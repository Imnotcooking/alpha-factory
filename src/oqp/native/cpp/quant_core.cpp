#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <cmath>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <map>
#include <numeric>
#include <limits>
#include <cstdint>
#include <deque>
#include <set>

namespace py = pybind11;

// ==========================================
// 0. NUMERICAL SOLVERS (Dynamic Programming)
// ==========================================

// The Abstract Bellman Engine
class BaseBellmanSolver {
protected:
    int total_time_steps;
    // 2D Grid: value_function[time_step][state_index]
    std::vector<std::vector<double>> value_function; 

public:
    BaseBellmanSolver(int T, int state_count) : total_time_steps(T) {
        value_function.resize(T + 1, std::vector<double>(state_count, -1e9));
    }
    virtual ~BaseBellmanSolver() = default;

    // The physical rules defined by child classes
    virtual int get_state_count() const = 0;
    virtual int state_to_index(double state) const = 0;
    virtual double state_from_index(int state_idx) const = 0;
    virtual std::vector<int> get_possible_action_indices(int state_idx, int t) = 0;
    virtual double calculate_immediate_reward(int state_idx, int action_idx, int t) = 0;
    
    // The Backward Induction Engine
    void solve() {
        // Base case: At expiration (T), remaining inventory has massive penalty
        for (int state_idx = 0; state_idx < get_state_count(); ++state_idx) {
            double state = state_from_index(state_idx);
            // If we didn't finish selling, apply a severe liquidation penalty
            value_function[total_time_steps][state_idx] = (state > 0) ? -1e9 : 0.0;
        }

        // Step backwards through time
        for (int t = total_time_steps - 1; t >= 0; --t) {
            for (int state_idx = 0; state_idx < get_state_count(); ++state_idx) {
                double max_value = -1e9; 
                std::vector<int> actions = get_possible_action_indices(state_idx, t);
                
                for (int action_idx : actions) {
                    double reward = calculate_immediate_reward(state_idx, action_idx, t);
                    
                    // Look up tomorrow's optimal value for the resulting state
                    int next_state_idx = state_idx - action_idx;
                    if (next_state_idx < 0 || next_state_idx >= get_state_count()) {
                        continue;
                    }
                    double future_val = value_function[t + 1][next_state_idx];
                    
                    double q_value = reward + future_val;
                    if (q_value > max_value) {
                        max_value = q_value;
                    }
                }
                value_function[t][state_idx] = max_value; 
            }
        }
    }

    // Helper to get the calculated optimal cost for starting state
    double get_optimal_value(double initial_state) {
        return value_function[0][state_to_index(initial_state)];
    }
};

class WaveletHurstEstimator {
public:
    // Helper function to calculate the variance of a vector
    static double calculate_variance(const std::vector<double>& data) {
        if (data.size() <= 1) return 0.0;
        double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        double var = 0.0;
        for (double x : data) {
            var += (x - mean) * (x - mean);
        }
        return var / (data.size() - 1);
    }

    // The Haar Transform and Regression Engine
    static double estimate_hurst(std::vector<double> prices) {
        // We need a power of 2 for Haar wavelets. 
        // If we don't have exactly 2^N prices, we pad or truncate.
        // For production, assume 'prices' comes in as an array of the last 64 or 128 ticks.
        
        std::vector<double> current_signal = prices;
        std::vector<double> log_scales;
        std::vector<double> log_variances;
        
        int scale = 1;

        // Keep chopping the signal in half until we run out of pairs
        while (current_signal.size() >= 2) {
            std::vector<double> next_signal;
            std::vector<double> detail_coeffs; // The "Noise" bucket
            
            // Step through array in pairs
            for (size_t i = 0; i < current_signal.size() - 1; i += 2) {
                // Average (Approximation)
                next_signal.push_back((current_signal[i] + current_signal[i+1]) / std::sqrt(2.0));
                // Difference (Detail)
                detail_coeffs.push_back((current_signal[i] - current_signal[i+1]) / std::sqrt(2.0));
            }
            
            // Calculate variance of the noise at this scale
            double variance = calculate_variance(detail_coeffs);
            
            // We only use levels with valid variance for the regression
            if (variance > 1e-10) {
                log_scales.push_back(std::log2(scale));
                log_variances.push_back(std::log2(variance));
            }
            
            // Move to the next level
            current_signal = next_signal;
            scale *= 2; 
        }

        // --- Linear Regression (Ordinary Least Squares) ---
        // We want the slope (m) of log_variances vs log_scales
        if (log_scales.size() < 2) return 0.5; // Fallback to Random Walk if not enough data

        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        int n = log_scales.size();
        
        for (int i = 0; i < n; ++i) {
            sum_x += log_scales[i];
            sum_y += log_variances[i];
            sum_xy += log_scales[i] * log_variances[i];
            sum_x2 += log_scales[i] * log_scales[i];
        }
        
        // OLS Slope Formula
        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        
        // Hurst Exponent Formula from Wavelet Variance Slope
        double H = (slope - 1.0) / 2.0;

        // Clamp H between bounds just in case of weird data
        return std::max(0.1, std::min(0.9, H));
    }
};

// The Concrete Almgren-Chriss Implementation
class AlmgrenChrissDP : public BaseBellmanSolver {
private:
    static constexpr int INVENTORY_STEPS = 10;
    double risk_aversion;   // lambda
    double asset_volatility; // sigma
    double impact_coeff;    // eta
    double total_inventory;

public:
    AlmgrenChrissDP(int T, double lambda, double vol, double eta, double inventory) 
        : BaseBellmanSolver(T, INVENTORY_STEPS + 1), risk_aversion(lambda), asset_volatility(vol), 
          impact_coeff(eta), total_inventory(inventory) {}

    int get_state_count() const override {
        return INVENTORY_STEPS + 1;
    }

    int state_to_index(double state) const override {
        if (total_inventory <= 0.0) return 0;
        int idx = static_cast<int>(std::llround((state / total_inventory) * INVENTORY_STEPS));
        return std::max(0, std::min(INVENTORY_STEPS, idx));
    }

    double state_from_index(int state_idx) const override {
        return (total_inventory / INVENTORY_STEPS) * state_idx;
    }

    std::vector<int> get_possible_action_indices(int state_idx, int t) override {
        // Can sell anywhere from 0 up to current state (inventory), in discrete chunks
        std::vector<int> actions;
        for (int action_idx = 0; action_idx <= state_idx; ++action_idx) {
            actions.push_back(action_idx);
        }
        return actions;
    }

    double calculate_immediate_reward(int state_idx, int action_idx, int t) override {
        double state = state_from_index(state_idx);
        double action = state_from_index(action_idx);
        if (action == 0.0 && state > 0.0) {
            // Holding inventory costs risk
            return -(risk_aversion * std::pow(state, 2) * std::pow(asset_volatility, 2));
        }
        
        // Square Root Impact: eta * sqrt(action)
        double impact_cost = impact_coeff * std::sqrt(action);
        
        // Variance Penalty for remaining inventory
        double remaining = state - action;
        double risk_penalty = risk_aversion * std::pow(remaining, 2) * std::pow(asset_volatility, 2);
        
        return -(impact_cost + risk_penalty); 
    }
};

// ==========================================
// 1. ABSTRACT INTERFACES (The Contracts)
// ==========================================
class ITCAModel {
public:
    virtual ~ITCAModel() = default;
    virtual double calculate_slippage(double trade_size_notional, double notional_volume, double price, double volatility, double hurst) = 0;
};

class IMarginModel {
public:
    virtual ~IMarginModel() = default;
    virtual bool check_liquidation(double equity, double total_exposure) = 0;
};

// ==========================================
// 2. CONCRETE CLASSES (Market Mechanics)
// ==========================================

class SquareRootTCA : public ITCAModel {
private:
    double base_bps;
    double impact_gamma;
public:
    SquareRootTCA(double bps, double gamma) : base_bps(bps), impact_gamma(gamma) {}
    
    // UPDATED SIGNATURE: Accepts vol and hurst to satisfy the parent interface, but doesn't use them
    double calculate_slippage(double trade_size_notional, double notional_volume, double price, double volatility, double hurst) override {
        double safe_vol = std::max(price * notional_volume, 1.0); // Prevent Div by Zero
        double participation_rate = std::min(trade_size_notional / safe_vol, 0.20);
        double market_impact = impact_gamma * std::sqrt(participation_rate);
        return trade_size_notional * (base_bps + market_impact);
    }
};

class CryptoOrderBookTCA : public ITCAModel {
private:
    double base_bps;
    double depth_penalty;
public:
    CryptoOrderBookTCA(double bps = 0.0005, double penalty = 2.0) : base_bps(bps), depth_penalty(penalty) {}
    
    // UPDATED SIGNATURE
    double calculate_slippage(double trade_size_notional, double notional_volume, double price, double volatility, double hurst) override {
        double safe_vol = std::max(price * notional_volume, 1.0); 
        double participation = trade_size_notional / safe_vol;
        
        double effective_bps = base_bps;
        if (participation > 0.05) { effective_bps *= depth_penalty; }
        return trade_size_notional * effective_bps;
    }
};

class StochasticTCAWrapper : public ITCAModel {
private:
    double risk_aversion;      
    double impact_coeff;       
    double hurst_sensitivity;  
    int execution_windows;     

public:
    StochasticTCAWrapper(double lambda = 1e-4, double eta = 0.1, double gamma = 2.0, int T = 60)
        : risk_aversion(lambda), impact_coeff(eta), hurst_sensitivity(gamma), execution_windows(T) {}

    double calculate_slippage(double trade_size_notional, double notional_volume, double price, double volatility, double hurst) override {
        if (trade_size_notional == 0.0 || price == 0.0) return 0.0;
        
        double total_inventory = std::abs(trade_size_notional) / price;
        double adjusted_volatility = std::max(volatility * (1.0 + hurst_sensitivity * (0.5 - hurst)), 1e-6);
        double dynamic_eta = impact_coeff / std::max(notional_volume, 1.0);

        AlmgrenChrissDP solver(execution_windows, risk_aversion, adjusted_volatility, dynamic_eta, total_inventory);
        solver.solve();

        double optimal_total_cost = -solver.get_optimal_value(total_inventory);
        double max_allowed_slippage = std::abs(trade_size_notional) * 0.05;
        return std::min(optimal_total_cost, max_allowed_slippage);
    }
};

class FuturesMargin : public IMarginModel {
private:
    double maintenance_req;
public:
    FuturesMargin(double req) : maintenance_req(req) {}
    bool check_liquidation(double equity, double total_exposure) override {
        double required_maintenance = total_exposure * equity * maintenance_req;
        return equity <= required_maintenance || equity <= 0;
    }
};

class EquitiesMargin : public IMarginModel {
private:
    double maintenance_req;
public:
    EquitiesMargin(double req = 0.25) : maintenance_req(req) {}
    bool check_liquidation(double equity, double total_exposure) override {
        double required_maintenance = total_exposure * equity * maintenance_req;
        return equity <= required_maintenance || equity <= 0;
    }
};


// ==========================================
// 3. THE EXECUTION ENGINE
// ==========================================
struct AssetState {
    double current_w = 0.0;
    double current_contracts = 0.0;
    double prev_price = -1.0; 
    int last_long_increase_date = std::numeric_limits<int>::min();
    int last_short_increase_date = std::numeric_limits<int>::min();
};

class ExecutionEngine {
private:
    std::shared_ptr<ITCAModel> tca_model;
    std::shared_ptr<IMarginModel> margin_model;
    
    double equity;
    double deadband;
    bool is_liquidated;
    std::unordered_map<int, AssetState> asset_states;
    double total_gross_w = 0.0; 
    
    bool enforce_price_limits;
    bool enforce_t1_settlement;
    double limit_board_pct = 0.099; 
    double fixed_slippage_ticks_per_side;

    static int sign_of(double value) {
        if (value > 0.0) return 1;
        if (value < 0.0) return -1;
        return 0;
    }

    static double fee_component(
        double weight_abs,
        double equity_value,
        double adjusted_price,
        int fee_type,
        double fee_rate
    ) {
        if (weight_abs <= 0.0 || equity_value <= 0.0 || adjusted_price <= 0.0) return 0.0;
        double notional = weight_abs * equity_value;
        double contracts = notional / adjusted_price;
        if (fee_type == 1) {
            return contracts * fee_rate;
        }
        return notional * fee_rate;
    }

    static double weight_to_contracts(double weight, double equity_value, double adjusted_price) {
        if (!std::isfinite(weight) || equity_value <= 0.0 || adjusted_price <= 0.0) return 0.0;
        return weight * equity_value / adjusted_price;
    }

    static double contracts_to_weight(double contracts, double equity_value, double adjusted_price) {
        if (!std::isfinite(contracts) || equity_value <= 0.0 || adjusted_price <= 0.0) return 0.0;
        return contracts * adjusted_price / equity_value;
    }

    static double round_contract_lots(double contracts) {
        if (!std::isfinite(contracts)) return 0.0;
        return std::round(contracts);
    }

    static double fixed_tick_slippage_cost(
        double contract_change_abs,
        double multiplier,
        double tick_size,
        double ticks_per_side
    ) {
        if (contract_change_abs <= 0.0 || multiplier <= 0.0 || tick_size <= 0.0 || ticks_per_side <= 0.0) {
            return 0.0;
        }
        return contract_change_abs * multiplier * tick_size * ticks_per_side;
    }

    static double fee_component_contracts(
        double contracts_abs,
        double adjusted_price,
        int fee_type,
        double fee_rate
    ) {
        if (contracts_abs <= 0.0 || adjusted_price <= 0.0) return 0.0;
        if (fee_type == 1) {
            return contracts_abs * fee_rate;
        }
        return contracts_abs * adjusted_price * fee_rate;
    }

    static double calculate_exchange_fee_contracts(
        double previous_contracts,
        double final_contracts,
        double adjusted_price,
        int fee_type,
        double fee_open,
        double fee_close_history,
        double fee_close_today,
        int date_id,
        const AssetState& state
    ) {
        int previous_sign = sign_of(previous_contracts);
        int final_sign = sign_of(final_contracts);
        double previous_abs = std::abs(previous_contracts);
        double final_abs = std::abs(final_contracts);

        double open_contracts = 0.0;
        double close_contracts = 0.0;
        bool close_today = false;

        if (previous_sign == 0) {
            open_contracts = final_abs;
        } else if (final_sign == 0) {
            close_contracts = previous_abs;
            close_today = previous_sign > 0
                ? state.last_long_increase_date == date_id
                : state.last_short_increase_date == date_id;
        } else if (previous_sign == final_sign) {
            if (final_abs > previous_abs) {
                open_contracts = final_abs - previous_abs;
            } else {
                close_contracts = previous_abs - final_abs;
                close_today = previous_sign > 0
                    ? state.last_long_increase_date == date_id
                    : state.last_short_increase_date == date_id;
            }
        } else {
            close_contracts = previous_abs;
            open_contracts = final_abs;
            close_today = previous_sign > 0
                ? state.last_long_increase_date == date_id
                : state.last_short_increase_date == date_id;
        }

        double close_fee_rate = close_today ? fee_close_today : fee_close_history;
        return fee_component_contracts(open_contracts, adjusted_price, fee_type, fee_open)
            + fee_component_contracts(close_contracts, adjusted_price, fee_type, close_fee_rate);
    }

    static double calculate_exchange_fee(
        double previous_w,
        double final_w,
        double equity_value,
        double adjusted_price,
        int fee_type,
        double fee_open,
        double fee_close_history,
        double fee_close_today,
        int date_id,
        const AssetState& state
    ) {
        int previous_sign = sign_of(previous_w);
        int final_sign = sign_of(final_w);
        double previous_abs = std::abs(previous_w);
        double final_abs = std::abs(final_w);

        double open_abs = 0.0;
        double close_abs = 0.0;
        bool close_today = false;

        if (previous_sign == 0) {
            open_abs = final_abs;
        } else if (final_sign == 0) {
            close_abs = previous_abs;
            close_today = previous_sign > 0
                ? state.last_long_increase_date == date_id
                : state.last_short_increase_date == date_id;
        } else if (previous_sign == final_sign) {
            if (final_abs > previous_abs) {
                open_abs = final_abs - previous_abs;
            } else {
                close_abs = previous_abs - final_abs;
                close_today = previous_sign > 0
                    ? state.last_long_increase_date == date_id
                    : state.last_short_increase_date == date_id;
            }
        } else {
            close_abs = previous_abs;
            open_abs = final_abs;
            close_today = previous_sign > 0
                ? state.last_long_increase_date == date_id
                : state.last_short_increase_date == date_id;
        }

        double close_fee_rate = close_today ? fee_close_today : fee_close_history;
        return fee_component(open_abs, equity_value, adjusted_price, fee_type, fee_open)
            + fee_component(close_abs, equity_value, adjusted_price, fee_type, close_fee_rate);
    }

    static double round_trip_fee_ticks(
        double adjusted_price,
        double multiplier,
        double tick_size,
        int fee_type,
        double fee_open,
        double fee_close_history,
        double fee_close_today
    ) {
        double tick_value = multiplier * tick_size;
        if (adjusted_price <= 0.0 || tick_value <= 0.0) return 0.0;
        double close_fee = std::max(fee_close_history, fee_close_today);
        if (fee_type == 1) {
            return (fee_open + close_fee) / tick_value;
        }
        return adjusted_price * (fee_open + close_fee) / tick_value;
    }

    static double round_trip_fee_bps(
        double adjusted_price,
        int fee_type,
        double fee_open,
        double fee_close_history,
        double fee_close_today
    ) {
        if (adjusted_price <= 0.0) return 0.0;
        double close_fee = std::max(fee_close_history, fee_close_today);
        if (fee_type == 1) {
            return (fee_open + close_fee) / adjusted_price * 10000.0;
        }
        return (fee_open + close_fee) * 10000.0;
    }

public:
    ExecutionEngine(std::shared_ptr<ITCAModel> tca, std::shared_ptr<IMarginModel> margin, 
                    double initial_capital, double deadband, 
                    bool price_limits = false, bool t1_settlement = false,
                    double fixed_slippage_ticks = 0.0)
        : tca_model(tca), margin_model(margin), equity(initial_capital), 
          deadband(deadband), is_liquidated(false), 
          enforce_price_limits(price_limits), enforce_t1_settlement(t1_settlement),
          fixed_slippage_ticks_per_side(fixed_slippage_ticks) {}

    // UPDATED SIGNATURE: Added volatilities and hursts arrays
    py::array_t<double> run_simulation(
        py::array_t<int> asset_ids,
        py::array_t<double> prices,
        py::array_t<double> target_weights,
        py::array_t<double> volumes,
        py::array_t<double> volatilities,
        py::array_t<double> hursts,
        py::array_t<int> date_ids = py::array_t<int>()) 
    {
        auto id_buf = asset_ids.request();
        auto p_buf = prices.request();
        auto w_buf = target_weights.request();
        auto v_buf = volumes.request();
        auto vol_buf = volatilities.request();
        auto h_buf = hursts.request();
        auto d_buf = date_ids.request();

        size_t N = p_buf.size;
        bool has_date_ids = static_cast<size_t>(d_buf.size) == N;
        if (enforce_t1_settlement && !has_date_ids) {
            throw std::runtime_error("ExecutionEngine requires date_ids when enforce_t1 is true.");
        }
        auto result = py::array_t<double>(N);
        auto res_buf = result.request();

        int* id_ptr = static_cast<int*>(id_buf.ptr);
        double* p_ptr = static_cast<double*>(p_buf.ptr);
        double* w_ptr = static_cast<double*>(w_buf.ptr);
        double* v_ptr = static_cast<double*>(v_buf.ptr);
        double* vol_ptr = static_cast<double*>(vol_buf.ptr);
        double* h_ptr = static_cast<double*>(h_buf.ptr);
        int* d_ptr = has_date_ids ? static_cast<int*>(d_buf.ptr) : nullptr;
        double* res_ptr = static_cast<double*>(res_buf.ptr);

        for (size_t i = 0; i < N; ++i) {
            if (is_liquidated || equity <= 0) { res_ptr[i] = 0.0; continue; }

            int asset_id = id_ptr[i];
            double price = p_ptr[i];
            double target_w = w_ptr[i];
            double volume = v_ptr[i];
            double vol = vol_ptr[i];
            double hurst = h_ptr[i];
            int date_id = has_date_ids ? d_ptr[i] : 0;

            AssetState& state = asset_states[asset_id];
            if (state.prev_price < 0.0) { state.prev_price = price; }

            double asset_return = 0.0;
            if (state.prev_price > 0.0) { asset_return = (price - state.prev_price) / state.prev_price; }
            equity += asset_return * state.current_w * equity;

            bool is_limit_locked = false;
            if (enforce_price_limits && std::abs(asset_return) >= limit_board_pct) { is_limit_locked = true; }

            double weight_change = std::abs(target_w - state.current_w);
            double final_w = state.current_w; 
            bool t1_locked = false;
            if (enforce_t1_settlement) {
                bool reducing_same_day_long =
                    state.current_w > 0.0 &&
                    target_w < state.current_w &&
                    state.last_long_increase_date == date_id;
                bool reducing_same_day_short =
                    state.current_w < 0.0 &&
                    target_w > state.current_w &&
                    state.last_short_increase_date == date_id;
                t1_locked = reducing_same_day_long || reducing_same_day_short;
            }

            if (!is_limit_locked && !t1_locked && weight_change > deadband) { final_w = target_w; }

            double actual_weight_change = std::abs(final_w - state.current_w);
            double previous_w = state.current_w;

            if (actual_weight_change > 0) {
                double trade_size_notional = actual_weight_change * equity;
                // UPDATED CALL: Pass vol and hurst to the polymorphic TCA model
                double tca_cost_notional = tca_model->calculate_slippage(trade_size_notional, volume, price, vol, hurst);
                equity -= tca_cost_notional;
            }

            total_gross_w += (std::abs(final_w) - std::abs(state.current_w));
            if (margin_model->check_liquidation(equity, total_gross_w)) {
                is_liquidated = true;
                equity = 0.0;
            }

            if (actual_weight_change > 0.0) {
                if (final_w > previous_w && final_w > 0.0) {
                    state.last_long_increase_date = date_id;
                }
                if (final_w < previous_w && final_w < 0.0) {
                    state.last_short_increase_date = date_id;
                }
            }

            state.current_w = final_w;
            state.prev_price = price;
            res_ptr[i] = equity;
        }
        return result;
    }

    py::dict run_simulation_with_costs(
        py::array_t<int> asset_ids,
        py::array_t<double> prices,
        py::array_t<double> target_weights,
        py::array_t<double> volumes,
        py::array_t<double> volatilities,
        py::array_t<double> hursts,
        py::array_t<int> date_ids,
        py::array_t<double> multipliers,
        py::array_t<int> fee_types,
        py::array_t<double> fee_open,
        py::array_t<double> fee_close_history,
        py::array_t<double> fee_close_today,
        py::array_t<double> tick_sizes = py::array_t<double>(),
        bool integer_lots = false)
    {
        auto id_buf = asset_ids.request();
        auto p_buf = prices.request();
        auto w_buf = target_weights.request();
        auto v_buf = volumes.request();
        auto vol_buf = volatilities.request();
        auto h_buf = hursts.request();
        auto d_buf = date_ids.request();
        auto m_buf = multipliers.request();
        auto ft_buf = fee_types.request();
        auto fo_buf = fee_open.request();
        auto fch_buf = fee_close_history.request();
        auto fct_buf = fee_close_today.request();
        auto ts_buf = tick_sizes.request();

        size_t N = p_buf.size;
        bool has_tick_sizes = static_cast<size_t>(ts_buf.size) == N;
        if (static_cast<size_t>(ts_buf.size) != 0 && !has_tick_sizes) {
            throw std::runtime_error("ExecutionEngine tick_sizes must be empty or equal length.");
        }
        if (static_cast<size_t>(id_buf.size) != N ||
            static_cast<size_t>(w_buf.size) != N ||
            static_cast<size_t>(v_buf.size) != N ||
            static_cast<size_t>(vol_buf.size) != N ||
            static_cast<size_t>(h_buf.size) != N ||
            static_cast<size_t>(d_buf.size) != N ||
            static_cast<size_t>(m_buf.size) != N ||
            static_cast<size_t>(ft_buf.size) != N ||
            static_cast<size_t>(fo_buf.size) != N ||
            static_cast<size_t>(fch_buf.size) != N ||
            static_cast<size_t>(fct_buf.size) != N) {
            throw std::runtime_error("ExecutionEngine cost simulation arrays must have equal length.");
        }

        auto equity_curve = py::array_t<double>(N);
        auto gross_equity_curve = py::array_t<double>(N);
        auto slippage_cost = py::array_t<double>(N);
        auto exchange_fee = py::array_t<double>(N);
        auto total_cost = py::array_t<double>(N);
        auto executed_weight = py::array_t<double>(N);
        auto trade_notional = py::array_t<double>(N);
        auto trade_contracts = py::array_t<double>(N);
        auto portfolio_leverage = py::array_t<double>(N);
        auto desired_contracts = py::array_t<double>(N);
        auto position_contracts = py::array_t<double>(N);
        auto rounding_error_weight = py::array_t<double>(N);
        auto one_lot_weight = py::array_t<double>(N);
        auto round_trip_fee_bps_arr = py::array_t<double>(N);
        auto round_trip_fee_ticks_arr = py::array_t<double>(N);
        auto lot_constrained = py::array_t<double>(N);
        auto fee_constrained = py::array_t<double>(N);

        int* id_ptr = static_cast<int*>(id_buf.ptr);
        double* p_ptr = static_cast<double*>(p_buf.ptr);
        double* w_ptr = static_cast<double*>(w_buf.ptr);
        double* v_ptr = static_cast<double*>(v_buf.ptr);
        double* vol_ptr = static_cast<double*>(vol_buf.ptr);
        double* h_ptr = static_cast<double*>(h_buf.ptr);
        int* d_ptr = static_cast<int*>(d_buf.ptr);
        double* m_ptr = static_cast<double*>(m_buf.ptr);
        int* ft_ptr = static_cast<int*>(ft_buf.ptr);
        double* fo_ptr = static_cast<double*>(fo_buf.ptr);
        double* fch_ptr = static_cast<double*>(fch_buf.ptr);
        double* fct_ptr = static_cast<double*>(fct_buf.ptr);
        double* ts_ptr = has_tick_sizes ? static_cast<double*>(ts_buf.ptr) : nullptr;

        double* eq_ptr = static_cast<double*>(equity_curve.request().ptr);
        double* gross_eq_ptr = static_cast<double*>(gross_equity_curve.request().ptr);
        double* slip_ptr = static_cast<double*>(slippage_cost.request().ptr);
        double* fee_ptr = static_cast<double*>(exchange_fee.request().ptr);
        double* total_cost_ptr = static_cast<double*>(total_cost.request().ptr);
        double* exec_w_ptr = static_cast<double*>(executed_weight.request().ptr);
        double* notional_ptr = static_cast<double*>(trade_notional.request().ptr);
        double* contracts_ptr = static_cast<double*>(trade_contracts.request().ptr);
        double* leverage_ptr = static_cast<double*>(portfolio_leverage.request().ptr);
        double* desired_contracts_ptr = static_cast<double*>(desired_contracts.request().ptr);
        double* position_contracts_ptr = static_cast<double*>(position_contracts.request().ptr);
        double* rounding_error_ptr = static_cast<double*>(rounding_error_weight.request().ptr);
        double* one_lot_weight_ptr = static_cast<double*>(one_lot_weight.request().ptr);
        double* fee_bps_ptr = static_cast<double*>(round_trip_fee_bps_arr.request().ptr);
        double* fee_ticks_ptr = static_cast<double*>(round_trip_fee_ticks_arr.request().ptr);
        double* lot_constrained_ptr = static_cast<double*>(lot_constrained.request().ptr);
        double* fee_constrained_ptr = static_cast<double*>(fee_constrained.request().ptr);

        for (size_t i = 0; i < N; ++i) {
            slip_ptr[i] = 0.0;
            fee_ptr[i] = 0.0;
            total_cost_ptr[i] = 0.0;
            notional_ptr[i] = 0.0;
            contracts_ptr[i] = 0.0;
            desired_contracts_ptr[i] = 0.0;
            position_contracts_ptr[i] = 0.0;
            rounding_error_ptr[i] = 0.0;
            one_lot_weight_ptr[i] = 0.0;
            fee_bps_ptr[i] = 0.0;
            fee_ticks_ptr[i] = 0.0;
            lot_constrained_ptr[i] = 0.0;
            fee_constrained_ptr[i] = 0.0;

            if (is_liquidated || equity <= 0) {
                eq_ptr[i] = 0.0;
                gross_eq_ptr[i] = 0.0;
                exec_w_ptr[i] = 0.0;
                leverage_ptr[i] = 0.0;
                continue;
            }

            int asset_id = id_ptr[i];
            double price = p_ptr[i];
            double target_w = w_ptr[i];
            double volume = v_ptr[i];
            double vol = vol_ptr[i];
            double hurst = h_ptr[i];
            int date_id = d_ptr[i];
            double multiplier = m_ptr[i];
            double tick_size = has_tick_sizes ? ts_ptr[i] : 1.0;

            AssetState& state = asset_states[asset_id];
            if (state.prev_price < 0.0) { state.prev_price = price; }

            double asset_return = 0.0;
            if (state.prev_price > 0.0) { asset_return = (price - state.prev_price) / state.prev_price; }
            equity += asset_return * state.current_w * equity;
            double gross_equity = equity;

            bool is_limit_locked = false;
            if (enforce_price_limits && std::abs(asset_return) >= limit_board_pct) { is_limit_locked = true; }

            double desired_contract_count = weight_to_contracts(target_w, equity, price);
            double rounded_contract_count = integer_lots
                ? round_contract_lots(desired_contract_count)
                : desired_contract_count;
            double rounded_target_w = integer_lots
                ? contracts_to_weight(rounded_contract_count, equity, price)
                : target_w;
            double lot_weight = price > 0.0 && equity > 0.0 ? price / equity : 0.0;
            double fee_ticks = round_trip_fee_ticks(
                price, multiplier, tick_size, ft_ptr[i], fo_ptr[i], fch_ptr[i], fct_ptr[i]
            );
            double fee_bps = round_trip_fee_bps(
                price, ft_ptr[i], fo_ptr[i], fch_ptr[i], fct_ptr[i]
            );

            desired_contracts_ptr[i] = desired_contract_count;
            one_lot_weight_ptr[i] = lot_weight;
            rounding_error_ptr[i] = integer_lots ? (rounded_target_w - target_w) : 0.0;
            fee_ticks_ptr[i] = fee_ticks;
            fee_bps_ptr[i] = fee_bps;
            lot_constrained_ptr[i] = (
                integer_lots &&
                std::abs(desired_contract_count) > 0.0 &&
                (std::abs(desired_contract_count) < 3.0 || std::abs(rounding_error_ptr[i]) > 0.25 * lot_weight)
            ) ? 1.0 : 0.0;
            fee_constrained_ptr[i] = (fee_ticks > 1.0 || fee_bps > 5.0) ? 1.0 : 0.0;

            double weight_change = std::abs(rounded_target_w - state.current_w);
            double final_w = state.current_w;
            double final_contracts = state.current_contracts;
            bool t1_locked = false;
            if (enforce_t1_settlement) {
                bool reducing_same_day_long =
                    state.current_w > 0.0 &&
                    rounded_target_w < state.current_w &&
                    state.last_long_increase_date == date_id;
                bool reducing_same_day_short =
                    state.current_w < 0.0 &&
                    rounded_target_w > state.current_w &&
                    state.last_short_increase_date == date_id;
                t1_locked = reducing_same_day_long || reducing_same_day_short;
            }

            if (!is_limit_locked && !t1_locked && weight_change > deadband) {
                final_w = rounded_target_w;
                final_contracts = rounded_contract_count;
            }

            double actual_weight_change = std::abs(final_w - state.current_w);
            double previous_w = state.current_w;
            double previous_contracts = state.current_contracts;
            double actual_contract_change = std::abs(final_contracts - previous_contracts);
            bool has_trade = integer_lots ? (actual_contract_change > 0.0) : (actual_weight_change > 0.0);

            if (has_trade) {
                double trade_size_notional = integer_lots
                    ? actual_contract_change * price
                    : actual_weight_change * equity;
                double traded_contracts = integer_lots
                    ? actual_contract_change
                    : (price > 0.0 ? trade_size_notional / price : 0.0);
                double model_slippage_cost = tca_model->calculate_slippage(trade_size_notional, volume, price, vol, hurst);
                double tick_slippage_cost = fixed_tick_slippage_cost(
                    traded_contracts,
                    multiplier,
                    tick_size,
                    fixed_slippage_ticks_per_side
                );
                double tca_cost_notional = model_slippage_cost + tick_slippage_cost;
                double fee_cost = integer_lots
                    ? calculate_exchange_fee_contracts(
                        previous_contracts,
                        final_contracts,
                        price,
                        ft_ptr[i],
                        fo_ptr[i],
                        fch_ptr[i],
                        fct_ptr[i],
                        date_id,
                        state
                    )
                    : calculate_exchange_fee(
                        previous_w,
                        final_w,
                        equity,
                        price,
                        ft_ptr[i],
                        fo_ptr[i],
                        fch_ptr[i],
                        fct_ptr[i],
                        date_id,
                        state
                    );

                notional_ptr[i] = trade_size_notional;
                contracts_ptr[i] = traded_contracts;
                slip_ptr[i] = tca_cost_notional;
                fee_ptr[i] = fee_cost;
                total_cost_ptr[i] = tca_cost_notional + fee_cost;
                equity -= total_cost_ptr[i];
            }

            total_gross_w += (std::abs(final_w) - std::abs(state.current_w));
            if (margin_model->check_liquidation(equity, total_gross_w)) {
                is_liquidated = true;
                equity = 0.0;
            }

            if (has_trade) {
                if (final_contracts > previous_contracts && final_contracts > 0.0) {
                    state.last_long_increase_date = date_id;
                }
                if (final_contracts < previous_contracts && final_contracts < 0.0) {
                    state.last_short_increase_date = date_id;
                }
            }

            state.current_w = final_w;
            state.current_contracts = integer_lots
                ? final_contracts
                : weight_to_contracts(final_w, equity, price);
            state.prev_price = price;
            eq_ptr[i] = equity;
            gross_eq_ptr[i] = gross_equity;
            exec_w_ptr[i] = final_w;
            position_contracts_ptr[i] = state.current_contracts;
            leverage_ptr[i] = total_gross_w;
        }

        py::dict result;
        result["equity_curve"] = equity_curve;
        result["gross_equity_curve"] = gross_equity_curve;
        result["slippage_cost"] = slippage_cost;
        result["exchange_fee"] = exchange_fee;
        result["total_cost"] = total_cost;
        result["executed_weight"] = executed_weight;
        result["trade_notional"] = trade_notional;
        result["trade_contracts"] = trade_contracts;
        result["portfolio_leverage"] = portfolio_leverage;
        result["desired_contracts"] = desired_contracts;
        result["position_contracts"] = position_contracts;
        result["rounding_error_weight"] = rounding_error_weight;
        result["one_lot_weight"] = one_lot_weight;
        result["round_trip_fee_bps"] = round_trip_fee_bps_arr;
        result["round_trip_fee_ticks"] = round_trip_fee_ticks_arr;
        result["is_lot_constrained"] = lot_constrained;
        result["is_fee_constrained"] = fee_constrained;
        return result;
    }

    py::dict run_simulation_with_costs_and_returns(
        py::array_t<int> asset_ids,
        py::array_t<double> prices,
        py::array_t<double> target_weights,
        py::array_t<double> period_returns,
        py::array_t<double> volumes,
        py::array_t<double> volatilities,
        py::array_t<double> hursts,
        py::array_t<int> time_ids,
        py::array_t<int> date_ids,
        py::array_t<double> multipliers,
        py::array_t<int> fee_types,
        py::array_t<double> fee_open,
        py::array_t<double> fee_close_history,
        py::array_t<double> fee_close_today,
        py::array_t<double> tick_sizes = py::array_t<double>(),
        bool integer_lots = false)
    {
        auto id_buf = asset_ids.request();
        auto p_buf = prices.request();
        auto w_buf = target_weights.request();
        auto r_buf = period_returns.request();
        auto v_buf = volumes.request();
        auto vol_buf = volatilities.request();
        auto h_buf = hursts.request();
        auto t_buf = time_ids.request();
        auto d_buf = date_ids.request();
        auto m_buf = multipliers.request();
        auto ft_buf = fee_types.request();
        auto fo_buf = fee_open.request();
        auto fch_buf = fee_close_history.request();
        auto fct_buf = fee_close_today.request();
        auto ts_buf = tick_sizes.request();

        size_t N = p_buf.size;
        bool has_tick_sizes = static_cast<size_t>(ts_buf.size) == N;
        if (static_cast<size_t>(ts_buf.size) != 0 && !has_tick_sizes) {
            throw std::runtime_error("ExecutionEngine tick_sizes must be empty or equal length.");
        }
        if (static_cast<size_t>(id_buf.size) != N ||
            static_cast<size_t>(w_buf.size) != N ||
            static_cast<size_t>(r_buf.size) != N ||
            static_cast<size_t>(v_buf.size) != N ||
            static_cast<size_t>(vol_buf.size) != N ||
            static_cast<size_t>(h_buf.size) != N ||
            static_cast<size_t>(t_buf.size) != N ||
            static_cast<size_t>(d_buf.size) != N ||
            static_cast<size_t>(m_buf.size) != N ||
            static_cast<size_t>(ft_buf.size) != N ||
            static_cast<size_t>(fo_buf.size) != N ||
            static_cast<size_t>(fch_buf.size) != N ||
            static_cast<size_t>(fct_buf.size) != N) {
            throw std::runtime_error("ExecutionEngine explicit-return arrays must have equal length.");
        }

        auto equity_curve = py::array_t<double>(N);
        auto gross_equity_curve = py::array_t<double>(N);
        auto slippage_cost = py::array_t<double>(N);
        auto exchange_fee = py::array_t<double>(N);
        auto total_cost = py::array_t<double>(N);
        auto executed_weight = py::array_t<double>(N);
        auto trade_notional = py::array_t<double>(N);
        auto trade_contracts = py::array_t<double>(N);
        auto portfolio_leverage = py::array_t<double>(N);
        auto desired_contracts = py::array_t<double>(N);
        auto position_contracts = py::array_t<double>(N);
        auto rounding_error_weight = py::array_t<double>(N);
        auto one_lot_weight = py::array_t<double>(N);
        auto round_trip_fee_bps_arr = py::array_t<double>(N);
        auto round_trip_fee_ticks_arr = py::array_t<double>(N);
        auto lot_constrained = py::array_t<double>(N);
        auto fee_constrained = py::array_t<double>(N);

        int* id_ptr = static_cast<int*>(id_buf.ptr);
        double* p_ptr = static_cast<double*>(p_buf.ptr);
        double* w_ptr = static_cast<double*>(w_buf.ptr);
        double* r_ptr = static_cast<double*>(r_buf.ptr);
        double* v_ptr = static_cast<double*>(v_buf.ptr);
        double* vol_ptr = static_cast<double*>(vol_buf.ptr);
        double* h_ptr = static_cast<double*>(h_buf.ptr);
        int* t_ptr = static_cast<int*>(t_buf.ptr);
        int* d_ptr = static_cast<int*>(d_buf.ptr);
        double* m_ptr = static_cast<double*>(m_buf.ptr);
        int* ft_ptr = static_cast<int*>(ft_buf.ptr);
        double* fo_ptr = static_cast<double*>(fo_buf.ptr);
        double* fch_ptr = static_cast<double*>(fch_buf.ptr);
        double* fct_ptr = static_cast<double*>(fct_buf.ptr);
        double* ts_ptr = has_tick_sizes ? static_cast<double*>(ts_buf.ptr) : nullptr;

        double* eq_ptr = static_cast<double*>(equity_curve.request().ptr);
        double* gross_eq_ptr = static_cast<double*>(gross_equity_curve.request().ptr);
        double* slip_ptr = static_cast<double*>(slippage_cost.request().ptr);
        double* fee_ptr = static_cast<double*>(exchange_fee.request().ptr);
        double* total_cost_ptr = static_cast<double*>(total_cost.request().ptr);
        double* exec_w_ptr = static_cast<double*>(executed_weight.request().ptr);
        double* notional_ptr = static_cast<double*>(trade_notional.request().ptr);
        double* contracts_ptr = static_cast<double*>(trade_contracts.request().ptr);
        double* leverage_ptr = static_cast<double*>(portfolio_leverage.request().ptr);
        double* desired_contracts_ptr = static_cast<double*>(desired_contracts.request().ptr);
        double* position_contracts_ptr = static_cast<double*>(position_contracts.request().ptr);
        double* rounding_error_ptr = static_cast<double*>(rounding_error_weight.request().ptr);
        double* one_lot_weight_ptr = static_cast<double*>(one_lot_weight.request().ptr);
        double* fee_bps_ptr = static_cast<double*>(round_trip_fee_bps_arr.request().ptr);
        double* fee_ticks_ptr = static_cast<double*>(round_trip_fee_ticks_arr.request().ptr);
        double* lot_constrained_ptr = static_cast<double*>(lot_constrained.request().ptr);
        double* fee_constrained_ptr = static_cast<double*>(fee_constrained.request().ptr);

        size_t i = 0;
        while (i < N) {
            size_t group_start = i;
            int time_id = t_ptr[i];
            size_t group_end = i + 1;
            while (group_end < N && t_ptr[group_end] == time_id) {
                ++group_end;
            }

            for (size_t j = group_start; j < group_end; ++j) {
                slip_ptr[j] = 0.0;
                fee_ptr[j] = 0.0;
                total_cost_ptr[j] = 0.0;
                notional_ptr[j] = 0.0;
                contracts_ptr[j] = 0.0;
                desired_contracts_ptr[j] = 0.0;
                position_contracts_ptr[j] = 0.0;
                rounding_error_ptr[j] = 0.0;
                one_lot_weight_ptr[j] = 0.0;
                fee_bps_ptr[j] = 0.0;
                fee_ticks_ptr[j] = 0.0;
                lot_constrained_ptr[j] = 0.0;
                fee_constrained_ptr[j] = 0.0;
            }

            if (is_liquidated || equity <= 0) {
                for (size_t j = group_start; j < group_end; ++j) {
                    eq_ptr[j] = 0.0;
                    gross_eq_ptr[j] = 0.0;
                    exec_w_ptr[j] = 0.0;
                    leverage_ptr[j] = 0.0;
                }
                i = group_end;
                continue;
            }

            double equity_before_group = equity;
            double group_total_cost = 0.0;
            double portfolio_return = 0.0;
            double projected_gross_w = total_gross_w;
            size_t group_size = group_end - group_start;
            std::vector<double> final_ws(group_size, 0.0);
            std::vector<double> previous_ws(group_size, 0.0);
            std::vector<double> final_contracts_vec(group_size, 0.0);
            std::vector<double> previous_contracts_vec(group_size, 0.0);
            std::vector<double> actual_changes(group_size, 0.0);
            std::vector<double> actual_contract_changes(group_size, 0.0);

            for (size_t j = group_start; j < group_end; ++j) {
                size_t offset = j - group_start;
                int asset_id = id_ptr[j];
                double price = p_ptr[j];
                double target_w = w_ptr[j];
                double asset_return = r_ptr[j];
                double volume = v_ptr[j];
                double vol = vol_ptr[j];
                double hurst = h_ptr[j];
                int date_id = d_ptr[j];
                double multiplier = m_ptr[j];
                double tick_size = has_tick_sizes ? ts_ptr[j] : 1.0;

                if (!std::isfinite(target_w)) { target_w = 0.0; }
                if (!std::isfinite(asset_return)) { asset_return = 0.0; }

                AssetState& state = asset_states[asset_id];
                if (state.prev_price < 0.0) { state.prev_price = price; }

                bool is_limit_locked = false;
                if (enforce_price_limits && std::abs(asset_return) >= limit_board_pct) { is_limit_locked = true; }

                double previous_w = state.current_w;
                double previous_contracts = state.current_contracts;
                double desired_contract_count = weight_to_contracts(target_w, equity_before_group, price);
                double rounded_contract_count = integer_lots
                    ? round_contract_lots(desired_contract_count)
                    : desired_contract_count;
                double rounded_target_w = integer_lots
                    ? contracts_to_weight(rounded_contract_count, equity_before_group, price)
                    : target_w;
                double lot_weight = price > 0.0 && equity_before_group > 0.0 ? price / equity_before_group : 0.0;
                double fee_ticks = round_trip_fee_ticks(
                    price, multiplier, tick_size, ft_ptr[j], fo_ptr[j], fch_ptr[j], fct_ptr[j]
                );
                double fee_bps = round_trip_fee_bps(
                    price, ft_ptr[j], fo_ptr[j], fch_ptr[j], fct_ptr[j]
                );

                desired_contracts_ptr[j] = desired_contract_count;
                one_lot_weight_ptr[j] = lot_weight;
                rounding_error_ptr[j] = integer_lots ? (rounded_target_w - target_w) : 0.0;
                fee_ticks_ptr[j] = fee_ticks;
                fee_bps_ptr[j] = fee_bps;
                lot_constrained_ptr[j] = (
                    integer_lots &&
                    std::abs(desired_contract_count) > 0.0 &&
                    (std::abs(desired_contract_count) < 3.0 || std::abs(rounding_error_ptr[j]) > 0.25 * lot_weight)
                ) ? 1.0 : 0.0;
                fee_constrained_ptr[j] = (fee_ticks > 1.0 || fee_bps > 5.0) ? 1.0 : 0.0;

                double weight_change = std::abs(rounded_target_w - previous_w);
                double final_w = previous_w;
                double final_contracts = previous_contracts;
                bool t1_locked = false;
                if (enforce_t1_settlement) {
                    bool reducing_same_day_long =
                        previous_w > 0.0 &&
                        rounded_target_w < previous_w &&
                        state.last_long_increase_date == date_id;
                    bool reducing_same_day_short =
                        previous_w < 0.0 &&
                        rounded_target_w > previous_w &&
                        state.last_short_increase_date == date_id;
                    t1_locked = reducing_same_day_long || reducing_same_day_short;
                }

                if (!is_limit_locked && !t1_locked && weight_change > deadband) {
                    final_w = rounded_target_w;
                    final_contracts = rounded_contract_count;
                }

                double actual_weight_change = std::abs(final_w - previous_w);
                double actual_contract_change = std::abs(final_contracts - previous_contracts);
                bool has_trade = integer_lots ? (actual_contract_change > 0.0) : (actual_weight_change > 0.0);
                if (has_trade) {
                    double trade_size_notional = integer_lots
                        ? actual_contract_change * price
                        : actual_weight_change * equity_before_group;
                    double traded_contracts = integer_lots
                        ? actual_contract_change
                        : (price > 0.0 ? trade_size_notional / price : 0.0);
                    double model_slippage_cost = tca_model->calculate_slippage(trade_size_notional, volume, price, vol, hurst);
                    double tick_slippage_cost = fixed_tick_slippage_cost(
                        traded_contracts,
                        multiplier,
                        tick_size,
                        fixed_slippage_ticks_per_side
                    );
                    double tca_cost_notional = model_slippage_cost + tick_slippage_cost;
                    double fee_cost = integer_lots
                        ? calculate_exchange_fee_contracts(
                            previous_contracts,
                            final_contracts,
                            price,
                            ft_ptr[j],
                            fo_ptr[j],
                            fch_ptr[j],
                            fct_ptr[j],
                            date_id,
                            state
                        )
                        : calculate_exchange_fee(
                            previous_w,
                            final_w,
                            equity_before_group,
                            price,
                            ft_ptr[j],
                            fo_ptr[j],
                            fch_ptr[j],
                            fct_ptr[j],
                            date_id,
                            state
                        );

                    notional_ptr[j] = trade_size_notional;
                    contracts_ptr[j] = traded_contracts;
                    slip_ptr[j] = tca_cost_notional;
                    fee_ptr[j] = fee_cost;
                    total_cost_ptr[j] = tca_cost_notional + fee_cost;
                    group_total_cost += total_cost_ptr[j];
                }

                projected_gross_w += (std::abs(final_w) - std::abs(previous_w));
                portfolio_return += final_w * asset_return;
                final_ws[offset] = final_w;
                previous_ws[offset] = previous_w;
                final_contracts_vec[offset] = final_contracts;
                previous_contracts_vec[offset] = previous_contracts;
                actual_changes[offset] = actual_weight_change;
                actual_contract_changes[offset] = actual_contract_change;
            }

            double equity_after_cost = equity_before_group - group_total_cost;
            if (equity_after_cost <= 0.0) {
                is_liquidated = true;
                equity = 0.0;
            } else {
                double gross_equity = equity_before_group * (1.0 + portfolio_return);
                equity = equity_after_cost * (1.0 + portfolio_return);
                total_gross_w = projected_gross_w;
                if (margin_model->check_liquidation(equity, total_gross_w)) {
                    is_liquidated = true;
                    equity = 0.0;
                }

                for (size_t j = group_start; j < group_end; ++j) {
                    size_t offset = j - group_start;
                    int asset_id = id_ptr[j];
                    double price = p_ptr[j];
                    int date_id = d_ptr[j];
                    AssetState& state = asset_states[asset_id];
                    double final_w = final_ws[offset];
                    double final_contracts = final_contracts_vec[offset];
                    double previous_contracts = previous_contracts_vec[offset];
                    bool has_trade = integer_lots
                        ? (actual_contract_changes[offset] > 0.0)
                        : (actual_changes[offset] > 0.0);

                    if (has_trade) {
                        if (final_contracts > previous_contracts && final_contracts > 0.0) {
                            state.last_long_increase_date = date_id;
                        }
                        if (final_contracts < previous_contracts && final_contracts < 0.0) {
                            state.last_short_increase_date = date_id;
                        }
                    }

                    state.current_w = final_w;
                    state.current_contracts = integer_lots
                        ? final_contracts
                        : weight_to_contracts(final_w, equity, price);
                    state.prev_price = price;
                    eq_ptr[j] = equity;
                    gross_eq_ptr[j] = gross_equity;
                    exec_w_ptr[j] = final_w;
                    position_contracts_ptr[j] = state.current_contracts;
                    leverage_ptr[j] = total_gross_w;
                }
            }

            if (is_liquidated) {
                for (size_t j = group_start; j < group_end; ++j) {
                    eq_ptr[j] = 0.0;
                    gross_eq_ptr[j] = 0.0;
                    exec_w_ptr[j] = final_ws[j - group_start];
                    leverage_ptr[j] = total_gross_w;
                }
            }

            i = group_end;
        }

        py::dict result;
        result["equity_curve"] = equity_curve;
        result["gross_equity_curve"] = gross_equity_curve;
        result["slippage_cost"] = slippage_cost;
        result["exchange_fee"] = exchange_fee;
        result["total_cost"] = total_cost;
        result["executed_weight"] = executed_weight;
        result["trade_notional"] = trade_notional;
        result["trade_contracts"] = trade_contracts;
        result["portfolio_leverage"] = portfolio_leverage;
        result["desired_contracts"] = desired_contracts;
        result["position_contracts"] = position_contracts;
        result["rounding_error_weight"] = rounding_error_weight;
        result["one_lot_weight"] = one_lot_weight;
        result["round_trip_fee_bps"] = round_trip_fee_bps_arr;
        result["round_trip_fee_ticks"] = round_trip_fee_ticks_arr;
        result["is_lot_constrained"] = lot_constrained;
        result["is_fee_constrained"] = fee_constrained;
        return result;
    }
};

// ==========================================
// 3. TICK PULSE RESEARCH KERNELS
// ==========================================
struct TickPulseCluster {
    int64_t first_index;
    int cluster_id;
    int cluster_size;
};

static bool is_valid_number(double value) {
    return std::isfinite(value);
}

class FenwickTree {
private:
    std::vector<int> tree;

public:
    explicit FenwickTree(size_t size) : tree(size + 1, 0) {}

    void add(size_t index, int delta) {
        for (size_t i = index + 1; i < tree.size(); i += i & -i) {
            tree[i] += delta;
        }
    }

    size_t find_by_order(int order_zero_based) const {
        int target = order_zero_based + 1;
        size_t idx = 0;
        size_t bit = 1;
        while ((bit << 1) < tree.size()) {
            bit <<= 1;
        }
        for (; bit > 0; bit >>= 1) {
            size_t next = idx + bit;
            if (next < tree.size() && tree[next] < target) {
                idx = next;
                target -= tree[next];
            }
        }
        return idx;
    }
};

static double fenwick_linear_quantile(
    const FenwickTree& counts,
    const std::vector<double>& values,
    int window_count,
    double percentile
) {
    if (window_count <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double q = std::max(0.0, std::min(1.0, percentile));
    double position = (static_cast<double>(window_count) - 1.0) * q;
    int lower_idx = static_cast<int>(std::floor(position));
    int upper_idx = static_cast<int>(std::ceil(position));
    double lower_value = values[counts.find_by_order(lower_idx)];
    if (lower_idx == upper_idx) {
        return lower_value;
    }
    double upper_value = values[counts.find_by_order(upper_idx)];
    double fraction = position - static_cast<double>(lower_idx);
    return lower_value + fraction * (upper_value - lower_value);
}

class RollingSumWindow {
private:
    std::deque<double> values;
    double running_sum = 0.0;
    int valid_count = 0;
    int window;

public:
    explicit RollingSumWindow(int window_size) : window(window_size) {}

    double push(double value, int min_periods) {
        values.push_back(value);
        if (is_valid_number(value)) {
            running_sum += value;
            valid_count += 1;
        }
        if (static_cast<int>(values.size()) > window) {
            double old = values.front();
            values.pop_front();
            if (is_valid_number(old)) {
                running_sum -= old;
                valid_count -= 1;
            }
        }
        if (valid_count >= min_periods) {
            return running_sum;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
};

class RollingMedianWindow {
private:
    std::deque<double> values;
    std::multiset<double> valid_values;
    int window;

public:
    explicit RollingMedianWindow(int window_size) : window(window_size) {}

    double push(double value, int min_periods) {
        values.push_back(value);
        if (is_valid_number(value)) {
            valid_values.insert(value);
        }
        if (static_cast<int>(values.size()) > window) {
            double old = values.front();
            values.pop_front();
            if (is_valid_number(old)) {
                auto it = valid_values.find(old);
                if (it != valid_values.end()) {
                    valid_values.erase(it);
                }
            }
        }
        if (static_cast<int>(valid_values.size()) < min_periods) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        size_t n = valid_values.size();
        auto upper = std::next(valid_values.begin(), static_cast<long>(n / 2));
        if (n % 2 == 1) {
            return *upper;
        }
        auto lower = std::prev(upper);
        return (*lower + *upper) / 2.0;
    }
};

static double sign_value(double value) {
    if (!is_valid_number(value)) return std::numeric_limits<double>::quiet_NaN();
    if (value > 0.0) return 1.0;
    if (value < 0.0) return -1.0;
    return 0.0;
}

static double median_from_values(std::vector<double> values) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

py::dict compute_tick_pulse_features_core(
    py::array_t<double, py::array::c_style | py::array::forcecast> last_price,
    py::array_t<double, py::array::c_style | py::array::forcecast> volume,
    py::array_t<double, py::array::c_style | py::array::forcecast> bid_price_1,
    py::array_t<double, py::array::c_style | py::array::forcecast> bid_volume_1,
    py::array_t<double, py::array::c_style | py::array::forcecast> ask_price_1,
    py::array_t<double, py::array::c_style | py::array::forcecast> ask_volume_1,
    py::array_t<double, py::array::c_style | py::array::forcecast> oi,
    py::array_t<double, py::array::c_style | py::array::forcecast> tick_size,
    py::array_t<int, py::array::c_style | py::array::forcecast> symbol_id,
    py::array_t<int, py::array::c_style | py::array::forcecast> session_id,
    int window,
    int min_periods
) {
    auto last_buf = last_price.request();
    auto volume_buf = volume.request();
    auto bid_buf = bid_price_1.request();
    auto bid_vol_buf = bid_volume_1.request();
    auto ask_buf = ask_price_1.request();
    auto ask_vol_buf = ask_volume_1.request();
    auto oi_buf = oi.request();
    auto tick_buf = tick_size.request();
    auto symbol_buf = symbol_id.request();
    auto session_buf = session_id.request();

    ssize_t raw_N = last_buf.size;
    if (
        volume_buf.size != raw_N ||
        bid_buf.size != raw_N ||
        bid_vol_buf.size != raw_N ||
        ask_buf.size != raw_N ||
        ask_vol_buf.size != raw_N ||
        oi_buf.size != raw_N ||
        tick_buf.size != raw_N ||
        symbol_buf.size != raw_N ||
        session_buf.size != raw_N
    ) {
        throw std::runtime_error("compute_tick_pulse_features_core input arrays must have equal length.");
    }
    if (window <= 0 || min_periods <= 0) {
        throw std::runtime_error("compute_tick_pulse_features_core received invalid rolling parameters.");
    }

    size_t N = static_cast<size_t>(raw_N);
    const double* last_ptr = static_cast<double*>(last_buf.ptr);
    const double* volume_ptr = static_cast<double*>(volume_buf.ptr);
    const double* bid_ptr = static_cast<double*>(bid_buf.ptr);
    const double* bid_vol_ptr = static_cast<double*>(bid_vol_buf.ptr);
    const double* ask_ptr = static_cast<double*>(ask_buf.ptr);
    const double* ask_vol_ptr = static_cast<double*>(ask_vol_buf.ptr);
    const double* oi_ptr = static_cast<double*>(oi_buf.ptr);
    const double* tick_ptr = static_cast<double*>(tick_buf.ptr);
    const int* symbol_ptr = static_cast<int*>(symbol_buf.ptr);
    const int* session_ptr = static_cast<int*>(session_buf.ptr);

    double nan = std::numeric_limits<double>::quiet_NaN();
    auto mid_price = py::array_t<double>(N);
    auto spread = py::array_t<double>(N);
    auto book_imbalance = py::array_t<double>(N);
    auto volume_delta_raw = py::array_t<double>(N);
    auto volume_delta = py::array_t<double>(N);
    auto oi_delta = py::array_t<double>(N);
    auto last_price_delta = py::array_t<double>(N);
    auto mid_price_delta = py::array_t<double>(N);
    auto mid_move_ticks = py::array_t<double>(N);
    auto trade_sign = py::array_t<double>(N);
    auto signed_volume = py::array_t<double>(N);
    auto rolling_signed_volume = py::array_t<double>(N);
    auto rolling_total_volume = py::array_t<double>(N);
    auto flow_imbalance = py::array_t<double>(N);
    auto rolling_pos_volume_median = py::array_t<double>(N);
    auto volume_intensity = py::array_t<double>(N);
    auto rolling_mid_move_ticks = py::array_t<double>(N);
    auto rolling_abs_tick_median = py::array_t<double>(N);
    auto price_shock = py::array_t<double>(N);
    auto pulse_score = py::array_t<double>(N);
    auto pulse_direction_code = py::array_t<int>(N);
    auto flow_price_aligned = py::array_t<bool>(N);
    auto book_flow_aligned = py::array_t<bool>(N);
    auto pulse_type_code = py::array_t<int>(N);

    double* mid = static_cast<double*>(mid_price.request().ptr);
    double* spr = static_cast<double*>(spread.request().ptr);
    double* book = static_cast<double*>(book_imbalance.request().ptr);
    double* vol_raw = static_cast<double*>(volume_delta_raw.request().ptr);
    double* vol_delta = static_cast<double*>(volume_delta.request().ptr);
    double* oi_d = static_cast<double*>(oi_delta.request().ptr);
    double* last_delta = static_cast<double*>(last_price_delta.request().ptr);
    double* mid_delta = static_cast<double*>(mid_price_delta.request().ptr);
    double* mid_ticks = static_cast<double*>(mid_move_ticks.request().ptr);
    double* sign = static_cast<double*>(trade_sign.request().ptr);
    double* signed_vol = static_cast<double*>(signed_volume.request().ptr);
    double* roll_signed = static_cast<double*>(rolling_signed_volume.request().ptr);
    double* roll_total = static_cast<double*>(rolling_total_volume.request().ptr);
    double* flow = static_cast<double*>(flow_imbalance.request().ptr);
    double* roll_pos_med = static_cast<double*>(rolling_pos_volume_median.request().ptr);
    double* vol_intensity = static_cast<double*>(volume_intensity.request().ptr);
    double* roll_mid = static_cast<double*>(rolling_mid_move_ticks.request().ptr);
    double* roll_abs_med = static_cast<double*>(rolling_abs_tick_median.request().ptr);
    double* shock = static_cast<double*>(price_shock.request().ptr);
    double* pulse = static_cast<double*>(pulse_score.request().ptr);
    int* pulse_dir = static_cast<int*>(pulse_direction_code.request().ptr);
    bool* flow_price = static_cast<bool*>(flow_price_aligned.request().ptr);
    bool* book_flow = static_cast<bool*>(book_flow_aligned.request().ptr);
    int* pulse_type = static_cast<int*>(pulse_type_code.request().ptr);

    for (size_t i = 0; i < N; ++i) {
        mid[i] = nan;
        spr[i] = nan;
        book[i] = 0.0;
        vol_raw[i] = 0.0;
        vol_delta[i] = 0.0;
        oi_d[i] = 0.0;
        last_delta[i] = 0.0;
        mid_delta[i] = 0.0;
        mid_ticks[i] = nan;
        sign[i] = 0.0;
        signed_vol[i] = 0.0;
        roll_signed[i] = nan;
        roll_total[i] = nan;
        flow[i] = 0.0;
        roll_pos_med[i] = nan;
        vol_intensity[i] = nan;
        roll_mid[i] = nan;
        roll_abs_med[i] = 0.0;
        shock[i] = nan;
        pulse[i] = nan;
        pulse_dir[i] = 0;
        flow_price[i] = false;
        book_flow[i] = false;
        pulse_type[i] = 3;
    }

    for (size_t i = 0; i < N; ++i) {
        if (is_valid_number(bid_ptr[i]) && is_valid_number(ask_ptr[i])) {
            mid[i] = (bid_ptr[i] + ask_ptr[i]) / 2.0;
            spr[i] = ask_ptr[i] - bid_ptr[i];
        }
        double depth = bid_vol_ptr[i] + ask_vol_ptr[i];
        if (is_valid_number(depth) && depth > 0.0) {
            book[i] = (bid_vol_ptr[i] - ask_vol_ptr[i]) / depth;
        } else {
            book[i] = 0.0;
        }
    }

    size_t start = 0;
    while (start < N) {
        size_t end = start + 1;
        while (end < N && symbol_ptr[end] == symbol_ptr[start]) {
            ++end;
        }

        double last_tick_rule = 0.0;
        size_t session_start = start;
        while (session_start < end) {
            size_t session_end = session_start + 1;
            while (
                session_end < end &&
                session_ptr[session_end] == session_ptr[session_start]
            ) {
                ++session_end;
            }

            for (size_t i = session_start; i < session_end; ++i) {
                if (i > session_start) {
                    vol_raw[i] = volume_ptr[i] - volume_ptr[i - 1];
                    oi_d[i] = oi_ptr[i] - oi_ptr[i - 1];
                    last_delta[i] = last_ptr[i] - last_ptr[i - 1];
                    mid_delta[i] = mid[i] - mid[i - 1];
                }
                if (vol_raw[i] > 0.0) {
                    vol_delta[i] = vol_raw[i];
                }
                if (is_valid_number(mid_delta[i]) && is_valid_number(tick_ptr[i]) && tick_ptr[i] != 0.0) {
                    mid_ticks[i] = mid_delta[i] / tick_ptr[i];
                }

                double tick_rule = sign_value(last_delta[i]);
                if (is_valid_number(tick_rule) && tick_rule != 0.0) {
                    last_tick_rule = tick_rule;
                } else {
                    tick_rule = last_tick_rule;
                }

                double prev_bid = (i > session_start) ? bid_ptr[i - 1] : nan;
                double prev_ask = (i > session_start) ? ask_ptr[i - 1] : nan;
                if (vol_delta[i] > 0.0 && is_valid_number(prev_ask) && last_ptr[i] >= prev_ask) {
                    sign[i] = 1.0;
                } else if (vol_delta[i] > 0.0 && is_valid_number(prev_bid) && last_ptr[i] <= prev_bid) {
                    sign[i] = -1.0;
                } else if (vol_delta[i] > 0.0) {
                    sign[i] = tick_rule;
                } else {
                    sign[i] = 0.0;
                }
                signed_vol[i] = sign[i] * vol_delta[i];
            }
            session_start = session_end;
        }

        std::vector<double> positive_volume_values;
        positive_volume_values.reserve(end - start);
        for (size_t i = start; i < end; ++i) {
            if (vol_delta[i] > 0.0) {
                positive_volume_values.push_back(vol_delta[i]);
            }
        }
        double fallback_median = median_from_values(positive_volume_values);
        if (!is_valid_number(fallback_median) || fallback_median == 0.0) {
            fallback_median = 1.0;
        }
        if (fallback_median < 1.0) {
            fallback_median = 1.0;
        }

        session_start = start;
        while (session_start < end) {
            size_t session_end = session_start + 1;
            while (
                session_end < end &&
                session_ptr[session_end] == session_ptr[session_start]
            ) {
                ++session_end;
            }

            RollingSumWindow signed_sum(window);
            RollingSumWindow total_sum(window);
            RollingMedianWindow positive_median(window);
            RollingSumWindow mid_move_sum(window);
            RollingMedianWindow abs_mid_median(window);

            for (size_t i = session_start; i < session_end; ++i) {
                roll_signed[i] = signed_sum.push(signed_vol[i], min_periods);
                roll_total[i] = total_sum.push(vol_delta[i], min_periods);
                if (is_valid_number(roll_total[i]) && roll_total[i] > 0.0) {
                    flow[i] = roll_signed[i] / roll_total[i];
                } else {
                    flow[i] = 0.0;
                }

                double positive_value = (vol_delta[i] > 0.0) ? vol_delta[i] : nan;
                roll_pos_med[i] = positive_median.push(positive_value, min_periods);
                if (!is_valid_number(roll_pos_med[i])) {
                    roll_pos_med[i] = fallback_median;
                }
                if (!is_valid_number(roll_pos_med[i])) {
                    roll_pos_med[i] = 1.0;
                }
                if (roll_pos_med[i] < 1.0) {
                    roll_pos_med[i] = 1.0;
                }
                vol_intensity[i] = vol_delta[i] / roll_pos_med[i];

                roll_mid[i] = mid_move_sum.push(mid_ticks[i], min_periods);
                double abs_mid = is_valid_number(mid_ticks[i]) ? std::abs(mid_ticks[i]) : nan;
                roll_abs_med[i] = abs_mid_median.push(abs_mid, min_periods);
                if (!is_valid_number(roll_abs_med[i])) {
                    roll_abs_med[i] = 0.0;
                }
                if (is_valid_number(roll_mid[i])) {
                    shock[i] = std::abs(roll_mid[i]) / (roll_abs_med[i] + 1.0);
                }

                double flow_strength = is_valid_number(flow[i]) ? std::min(1.0, std::max(0.0, std::abs(flow[i]))) : 0.0;
                double book_strength = is_valid_number(book[i]) ? std::min(1.0, std::max(0.0, std::abs(book[i]))) : 0.0;
                double clipped_volume = is_valid_number(vol_intensity[i]) ? std::max(0.0, vol_intensity[i]) : 0.0;
                double volume_strength = std::log1p(clipped_volume) / std::log1p(10.0);
                volume_strength = std::min(1.0, std::max(0.0, volume_strength));
                double price_strength = is_valid_number(shock[i]) ? shock[i] / 3.0 : 0.0;
                price_strength = std::min(1.0, std::max(0.0, price_strength));
                pulse[i] = (
                    0.40 * flow_strength +
                    0.25 * book_strength +
                    0.20 * volume_strength +
                    0.15 * price_strength
                );

                double direction_seed = (std::abs(flow[i]) >= 0.10) ? flow[i] : book[i];
                double direction_sign = sign_value(direction_seed);
                if (direction_sign > 0.0) {
                    pulse_dir[i] = 1;
                } else if (direction_sign < 0.0) {
                    pulse_dir[i] = -1;
                } else {
                    pulse_dir[i] = 0;
                }

                double flow_dir = sign_value(flow[i]);
                double book_dir = sign_value(book[i]);
                double price_dir = sign_value(roll_mid[i]);
                flow_price[i] = is_valid_number(flow_dir) && is_valid_number(price_dir) && (flow_dir * price_dir) > 0.0;
                book_flow[i] = is_valid_number(book_dir) && is_valid_number(flow_dir) && (book_dir * flow_dir) > 0.0;
                if (flow_price[i] && book_flow[i]) {
                    pulse_type[i] = 0;
                } else if (flow_price[i] && !book_flow[i]) {
                    pulse_type[i] = 1;
                } else if (!flow_price[i] && book_flow[i]) {
                    pulse_type[i] = 2;
                } else {
                    pulse_type[i] = 3;
                }
            }
            session_start = session_end;
        }

        start = end;
    }

    py::dict result;
    result["mid_price"] = mid_price;
    result["spread"] = spread;
    result["book_imbalance"] = book_imbalance;
    result["volume_delta_raw"] = volume_delta_raw;
    result["volume_delta"] = volume_delta;
    result["oi_delta"] = oi_delta;
    result["last_price_delta"] = last_price_delta;
    result["mid_price_delta"] = mid_price_delta;
    result["mid_move_ticks"] = mid_move_ticks;
    result["trade_sign"] = trade_sign;
    result["signed_volume"] = signed_volume;
    result["rolling_signed_volume"] = rolling_signed_volume;
    result["rolling_total_volume"] = rolling_total_volume;
    result["flow_imbalance"] = flow_imbalance;
    result["rolling_pos_volume_median"] = rolling_pos_volume_median;
    result["volume_intensity"] = volume_intensity;
    result["rolling_mid_move_ticks"] = rolling_mid_move_ticks;
    result["rolling_abs_tick_median"] = rolling_abs_tick_median;
    result["price_shock"] = price_shock;
    result["pulse_score"] = pulse_score;
    result["pulse_direction_code"] = pulse_direction_code;
    result["flow_price_aligned"] = flow_price_aligned;
    result["book_flow_aligned"] = book_flow_aligned;
    result["pulse_type_code"] = pulse_type_code;
    return result;
}

py::dict compute_tick_rtv_pipeline(
    py::array_t<double, py::array::c_style | py::array::forcecast> mid_price,
    py::array_t<double, py::array::c_style | py::array::forcecast> tick_size,
    py::array_t<int, py::array::c_style | py::array::forcecast> symbol_id,
    py::array_t<int, py::array::c_style | py::array::forcecast> session_id,
    int horizon_ticks,
    int fast_window,
    int slow_window,
    double percentile,
    int min_periods,
    double min_fast_move_ticks,
    double min_success_ticks,
    bool fade,
    int gap_ticks
) {
    auto mid_buf = mid_price.request();
    auto tick_buf = tick_size.request();
    auto symbol_buf = symbol_id.request();
    auto session_buf = session_id.request();

    ssize_t raw_N = mid_buf.size;
    if (tick_buf.size != raw_N || symbol_buf.size != raw_N || session_buf.size != raw_N) {
        throw std::runtime_error("compute_tick_rtv_pipeline input arrays must have equal length.");
    }
    size_t N = static_cast<size_t>(raw_N);
    if (horizon_ticks < 0 || fast_window < 0 || slow_window <= 0 || min_periods <= 0 || gap_ticks < 0) {
        throw std::runtime_error("compute_tick_rtv_pipeline received invalid window/horizon parameters.");
    }

    const double* mid_ptr = static_cast<double*>(mid_buf.ptr);
    const double* tick_ptr = static_cast<double*>(tick_buf.ptr);
    const int* symbol_ptr = static_cast<int*>(symbol_buf.ptr);
    const int* session_ptr = static_cast<int*>(session_buf.ptr);

    double nan = std::numeric_limits<double>::quiet_NaN();
    auto future_mid_price = py::array_t<double>(N);
    auto future_move_ticks = py::array_t<double>(N);
    auto rtv_fast_move_ticks = py::array_t<double>(N);
    auto rtv_abs_move_ticks = py::array_t<double>(N);
    auto rtv_threshold_ticks = py::array_t<double>(N);
    auto rtv_threshold_ratio = py::array_t<double>(N);
    auto rtv_direction_code = py::array_t<int>(N);
    auto expected_direction_code = py::array_t<int>(N);
    auto hypothesis_signal = py::array_t<bool>(N);
    auto is_correct = py::array_t<bool>(N);

    auto future_mid_buf = future_mid_price.request();
    auto future_move_buf = future_move_ticks.request();
    auto fast_buf = rtv_fast_move_ticks.request();
    auto abs_buf = rtv_abs_move_ticks.request();
    auto threshold_buf = rtv_threshold_ticks.request();
    auto ratio_buf = rtv_threshold_ratio.request();
    auto direction_buf = rtv_direction_code.request();
    auto expected_buf = expected_direction_code.request();
    auto signal_buf = hypothesis_signal.request();
    auto correct_buf = is_correct.request();

    double* future_mid = static_cast<double*>(future_mid_buf.ptr);
    double* future_move = static_cast<double*>(future_move_buf.ptr);
    double* fast_move = static_cast<double*>(fast_buf.ptr);
    double* abs_move = static_cast<double*>(abs_buf.ptr);
    double* threshold = static_cast<double*>(threshold_buf.ptr);
    double* ratio = static_cast<double*>(ratio_buf.ptr);
    int* direction = static_cast<int*>(direction_buf.ptr);
    int* expected = static_cast<int*>(expected_buf.ptr);
    bool* signal = static_cast<bool*>(signal_buf.ptr);
    bool* correct = static_cast<bool*>(correct_buf.ptr);

    for (size_t i = 0; i < N; ++i) {
        future_mid[i] = nan;
        future_move[i] = nan;
        fast_move[i] = nan;
        abs_move[i] = nan;
        threshold[i] = nan;
        ratio[i] = nan;
        direction[i] = 0;
        expected[i] = 0;
        signal[i] = false;
        correct[i] = false;
    }

    std::vector<TickPulseCluster> clusters;
    int cluster_id = 0;
    size_t start = 0;
    while (start < N) {
        size_t end = start + 1;
        while (
            end < N &&
            symbol_ptr[end] == symbol_ptr[start] &&
            session_ptr[end] == session_ptr[start]
        ) {
            ++end;
        }

        for (size_t i = start; i < end; ++i) {
            size_t local_pos = i - start;
            if (fast_window == 0) {
                fast_move[i] = 0.0;
            } else if (local_pos >= static_cast<size_t>(fast_window)) {
                size_t prev = i - static_cast<size_t>(fast_window);
                if (
                    is_valid_number(mid_ptr[i]) &&
                    is_valid_number(mid_ptr[prev]) &&
                    is_valid_number(tick_ptr[i]) &&
                    tick_ptr[i] != 0.0
                ) {
                    fast_move[i] = (mid_ptr[i] - mid_ptr[prev]) / tick_ptr[i];
                }
            }

            if (is_valid_number(fast_move[i])) {
                abs_move[i] = std::abs(fast_move[i]);
                if (fast_move[i] > 0.0) {
                    direction[i] = 1;
                } else if (fast_move[i] < 0.0) {
                    direction[i] = -1;
                }
            }

            if (horizon_ticks == 0) {
                future_mid[i] = mid_ptr[i];
            } else if (i + static_cast<size_t>(horizon_ticks) < end) {
                size_t future_idx = i + static_cast<size_t>(horizon_ticks);
                future_mid[i] = mid_ptr[future_idx];
            }

            if (
                is_valid_number(future_mid[i]) &&
                is_valid_number(mid_ptr[i]) &&
                is_valid_number(tick_ptr[i]) &&
                tick_ptr[i] != 0.0
            ) {
                future_move[i] = (future_mid[i] - mid_ptr[i]) / tick_ptr[i];
            }
        }

        std::vector<double> unique_abs_moves;
        unique_abs_moves.reserve(end - start);
        for (size_t i = start; i < end; ++i) {
            if (is_valid_number(abs_move[i])) {
                unique_abs_moves.push_back(abs_move[i]);
            }
        }
        std::sort(unique_abs_moves.begin(), unique_abs_moves.end());
        unique_abs_moves.erase(
            std::unique(unique_abs_moves.begin(), unique_abs_moves.end()),
            unique_abs_moves.end()
        );
        FenwickTree rolling_counts(unique_abs_moves.size());
        int rolling_count = 0;
        for (size_t i = start; i < end; ++i) {
            size_t local_pos = i - start;
            if (rolling_count >= min_periods) {
                threshold[i] = fenwick_linear_quantile(
                    rolling_counts,
                    unique_abs_moves,
                    rolling_count,
                    percentile
                );
            }

            if (
                is_valid_number(abs_move[i]) &&
                is_valid_number(threshold[i]) &&
                threshold[i] != 0.0
            ) {
                ratio[i] = abs_move[i] / threshold[i];
            }

            if (fade) {
                expected[i] = -direction[i];
            } else {
                expected[i] = direction[i];
            }

            bool valid_signal = (
                is_valid_number(threshold[i]) &&
                is_valid_number(fast_move[i]) &&
                direction[i] != 0 &&
                is_valid_number(abs_move[i]) &&
                abs_move[i] >= min_fast_move_ticks &&
                abs_move[i] >= threshold[i]
            );
            signal[i] = valid_signal;

            if (expected[i] == 1 && is_valid_number(future_move[i])) {
                correct[i] = future_move[i] >= min_success_ticks;
            } else if (expected[i] == -1 && is_valid_number(future_move[i])) {
                correct[i] = future_move[i] <= -min_success_ticks;
            }

            if (is_valid_number(abs_move[i])) {
                auto it = std::lower_bound(unique_abs_moves.begin(), unique_abs_moves.end(), abs_move[i]);
                rolling_counts.add(static_cast<size_t>(it - unique_abs_moves.begin()), 1);
                rolling_count += 1;
            }
            if (local_pos + 1 > static_cast<size_t>(slow_window)) {
                size_t remove_idx = i - static_cast<size_t>(slow_window);
                if (is_valid_number(abs_move[remove_idx])) {
                    auto it = std::lower_bound(unique_abs_moves.begin(), unique_abs_moves.end(), abs_move[remove_idx]);
                    rolling_counts.add(static_cast<size_t>(it - unique_abs_moves.begin()), -1);
                    rolling_count -= 1;
                }
            }
        }

        bool has_open_cluster = false;
        TickPulseCluster open_cluster{0, 0, 0};
        int64_t previous_event_pos = -1;
        for (size_t i = start; i < end; ++i) {
            bool valid_candidate = signal[i] && is_valid_number(future_move[i]);
            if (!valid_candidate) {
                continue;
            }
            int64_t local_pos = static_cast<int64_t>(i - start);
            bool new_cluster = (
                !has_open_cluster ||
                (local_pos - previous_event_pos) > static_cast<int64_t>(gap_ticks)
            );
            if (new_cluster) {
                if (has_open_cluster) {
                    clusters.push_back(open_cluster);
                }
                open_cluster = TickPulseCluster{static_cast<int64_t>(i), cluster_id, 1};
                ++cluster_id;
                has_open_cluster = true;
            } else {
                open_cluster.cluster_size += 1;
            }
            previous_event_pos = local_pos;
        }
        if (has_open_cluster) {
            clusters.push_back(open_cluster);
        }

        start = end;
    }

    auto candidate_indices = py::array_t<int64_t>(clusters.size());
    auto event_cluster_id = py::array_t<int>(clusters.size());
    auto event_cluster_size = py::array_t<int>(clusters.size());
    auto candidate_buf = candidate_indices.request();
    auto cluster_id_buf = event_cluster_id.request();
    auto cluster_size_buf = event_cluster_size.request();
    int64_t* candidate_ptr = static_cast<int64_t*>(candidate_buf.ptr);
    int* cluster_id_ptr = static_cast<int*>(cluster_id_buf.ptr);
    int* cluster_size_ptr = static_cast<int*>(cluster_size_buf.ptr);
    for (size_t i = 0; i < clusters.size(); ++i) {
        candidate_ptr[i] = clusters[i].first_index;
        cluster_id_ptr[i] = clusters[i].cluster_id;
        cluster_size_ptr[i] = clusters[i].cluster_size;
    }

    py::dict result;
    result["future_mid_price"] = future_mid_price;
    result["future_move_ticks"] = future_move_ticks;
    result["rtv_fast_move_ticks"] = rtv_fast_move_ticks;
    result["rtv_abs_move_ticks"] = rtv_abs_move_ticks;
    result["rtv_threshold_ticks"] = rtv_threshold_ticks;
    result["rtv_threshold_ratio"] = rtv_threshold_ratio;
    result["rtv_direction_code"] = rtv_direction_code;
    result["expected_direction_code"] = expected_direction_code;
    result["hypothesis_signal"] = hypothesis_signal;
    result["is_correct"] = is_correct;
    result["candidate_indices"] = candidate_indices;
    result["event_cluster_id"] = event_cluster_id;
    result["event_cluster_size"] = event_cluster_size;
    return result;
}

py::dict compute_tick_heuristic_pipeline(
    py::array_t<double, py::array::c_style | py::array::forcecast> mid_price,
    py::array_t<double, py::array::c_style | py::array::forcecast> tick_size,
    py::array_t<double, py::array::c_style | py::array::forcecast> flow_imbalance,
    py::array_t<double, py::array::c_style | py::array::forcecast> book_imbalance,
    py::array_t<double, py::array::c_style | py::array::forcecast> volume_intensity,
    py::array_t<double, py::array::c_style | py::array::forcecast> rolling_mid_move_ticks,
    py::array_t<double, py::array::c_style | py::array::forcecast> price_shock,
    py::array_t<int, py::array::c_style | py::array::forcecast> symbol_id,
    py::array_t<int, py::array::c_style | py::array::forcecast> session_id,
    int horizon_ticks,
    int hypothesis_code,
    double min_success_ticks,
    double flow_sell_max,
    double book_buy_min,
    double book_sell_max,
    double breakdown_book_max,
    double volume_burst_min,
    double breakdown_volume_burst_min,
    double rolling_mid_up_min,
    double rolling_mid_down_max,
    double breakdown_rolling_mid_max,
    double price_shock_min,
    int gap_ticks
) {
    auto mid_buf = mid_price.request();
    auto tick_buf = tick_size.request();
    auto flow_buf = flow_imbalance.request();
    auto book_buf = book_imbalance.request();
    auto volume_buf = volume_intensity.request();
    auto rolling_buf = rolling_mid_move_ticks.request();
    auto shock_buf = price_shock.request();
    auto symbol_buf = symbol_id.request();
    auto session_buf = session_id.request();

    ssize_t raw_N = mid_buf.size;
    if (
        tick_buf.size != raw_N ||
        flow_buf.size != raw_N ||
        book_buf.size != raw_N ||
        volume_buf.size != raw_N ||
        rolling_buf.size != raw_N ||
        shock_buf.size != raw_N ||
        symbol_buf.size != raw_N ||
        session_buf.size != raw_N
    ) {
        throw std::runtime_error("compute_tick_heuristic_pipeline input arrays must have equal length.");
    }
    if (horizon_ticks < 0 || gap_ticks < 0 || hypothesis_code < 0 || hypothesis_code > 2) {
        throw std::runtime_error("compute_tick_heuristic_pipeline received invalid parameters.");
    }

    size_t N = static_cast<size_t>(raw_N);
    const double* mid_ptr = static_cast<double*>(mid_buf.ptr);
    const double* tick_ptr = static_cast<double*>(tick_buf.ptr);
    const double* flow_ptr = static_cast<double*>(flow_buf.ptr);
    const double* book_ptr = static_cast<double*>(book_buf.ptr);
    const double* volume_ptr = static_cast<double*>(volume_buf.ptr);
    const double* rolling_ptr = static_cast<double*>(rolling_buf.ptr);
    const double* shock_ptr = static_cast<double*>(shock_buf.ptr);
    const int* symbol_ptr = static_cast<int*>(symbol_buf.ptr);
    const int* session_ptr = static_cast<int*>(session_buf.ptr);

    double nan = std::numeric_limits<double>::quiet_NaN();
    auto future_mid_price = py::array_t<double>(N);
    auto future_move_ticks = py::array_t<double>(N);
    auto expected_direction_code = py::array_t<int>(N);
    auto criterion_flow = py::array_t<bool>(N);
    auto criterion_book = py::array_t<bool>(N);
    auto criterion_price_resilience = py::array_t<bool>(N);
    auto criterion_volume_burst = py::array_t<bool>(N);
    auto criterion_price_shock = py::array_t<bool>(N);
    auto hypothesis_signal = py::array_t<bool>(N);
    auto is_correct = py::array_t<bool>(N);

    auto future_mid_buf = future_mid_price.request();
    auto future_move_buf = future_move_ticks.request();
    auto expected_buf = expected_direction_code.request();
    auto flow_criterion_buf = criterion_flow.request();
    auto book_criterion_buf = criterion_book.request();
    auto resilience_criterion_buf = criterion_price_resilience.request();
    auto volume_criterion_buf = criterion_volume_burst.request();
    auto shock_criterion_buf = criterion_price_shock.request();
    auto signal_buf = hypothesis_signal.request();
    auto correct_buf = is_correct.request();

    double* future_mid = static_cast<double*>(future_mid_buf.ptr);
    double* future_move = static_cast<double*>(future_move_buf.ptr);
    int* expected = static_cast<int*>(expected_buf.ptr);
    bool* flow_ok = static_cast<bool*>(flow_criterion_buf.ptr);
    bool* book_ok = static_cast<bool*>(book_criterion_buf.ptr);
    bool* resilience_ok = static_cast<bool*>(resilience_criterion_buf.ptr);
    bool* volume_ok = static_cast<bool*>(volume_criterion_buf.ptr);
    bool* shock_ok = static_cast<bool*>(shock_criterion_buf.ptr);
    bool* signal = static_cast<bool*>(signal_buf.ptr);
    bool* correct = static_cast<bool*>(correct_buf.ptr);

    for (size_t i = 0; i < N; ++i) {
        future_mid[i] = nan;
        future_move[i] = nan;
        expected[i] = (hypothesis_code == 0) ? 1 : -1;
        flow_ok[i] = false;
        book_ok[i] = false;
        resilience_ok[i] = false;
        volume_ok[i] = false;
        shock_ok[i] = false;
        signal[i] = false;
        correct[i] = false;
    }

    std::vector<TickPulseCluster> clusters;
    int cluster_id = 0;
    size_t start = 0;
    while (start < N) {
        size_t end = start + 1;
        while (
            end < N &&
            symbol_ptr[end] == symbol_ptr[start] &&
            session_ptr[end] == session_ptr[start]
        ) {
            ++end;
        }

        for (size_t i = start; i < end; ++i) {
            if (horizon_ticks == 0) {
                future_mid[i] = mid_ptr[i];
            } else if (i + static_cast<size_t>(horizon_ticks) < end) {
                future_mid[i] = mid_ptr[i + static_cast<size_t>(horizon_ticks)];
            }

            if (
                is_valid_number(future_mid[i]) &&
                is_valid_number(mid_ptr[i]) &&
                is_valid_number(tick_ptr[i]) &&
                tick_ptr[i] != 0.0
            ) {
                future_move[i] = (future_mid[i] - mid_ptr[i]) / tick_ptr[i];
            }

            if (hypothesis_code == 2) {
                flow_ok[i] = is_valid_number(flow_ptr[i]) && flow_ptr[i] <= flow_sell_max;
                book_ok[i] = is_valid_number(book_ptr[i]) && book_ptr[i] <= breakdown_book_max;
                volume_ok[i] = is_valid_number(volume_ptr[i]) && volume_ptr[i] >= breakdown_volume_burst_min;
                resilience_ok[i] = is_valid_number(rolling_ptr[i]) && rolling_ptr[i] <= breakdown_rolling_mid_max;
                shock_ok[i] = is_valid_number(shock_ptr[i]) && shock_ptr[i] >= price_shock_min;
                signal[i] = flow_ok[i] && book_ok[i] && volume_ok[i] && resilience_ok[i] && shock_ok[i];
            } else if (hypothesis_code == 1) {
                flow_ok[i] = is_valid_number(flow_ptr[i]) && flow_ptr[i] <= flow_sell_max;
                book_ok[i] = is_valid_number(book_ptr[i]) && book_ptr[i] <= book_sell_max;
                volume_ok[i] = is_valid_number(volume_ptr[i]) && volume_ptr[i] >= volume_burst_min;
                resilience_ok[i] = is_valid_number(rolling_ptr[i]) && rolling_ptr[i] <= rolling_mid_down_max;
                signal[i] = flow_ok[i] && book_ok[i] && volume_ok[i] && resilience_ok[i];
            } else {
                flow_ok[i] = is_valid_number(flow_ptr[i]) && flow_ptr[i] <= flow_sell_max;
                book_ok[i] = is_valid_number(book_ptr[i]) && book_ptr[i] >= book_buy_min;
                volume_ok[i] = is_valid_number(volume_ptr[i]) && volume_ptr[i] >= volume_burst_min;
                resilience_ok[i] = is_valid_number(rolling_ptr[i]) && rolling_ptr[i] >= rolling_mid_up_min;
                signal[i] = flow_ok[i] && book_ok[i] && volume_ok[i] && resilience_ok[i];
            }

            if (expected[i] == 1 && is_valid_number(future_move[i])) {
                correct[i] = future_move[i] >= min_success_ticks;
            } else if (expected[i] == -1 && is_valid_number(future_move[i])) {
                correct[i] = future_move[i] <= -min_success_ticks;
            }
        }

        bool has_open_cluster = false;
        TickPulseCluster open_cluster{0, 0, 0};
        int64_t previous_event_pos = -1;
        for (size_t i = start; i < end; ++i) {
            bool valid_candidate = signal[i] && is_valid_number(future_move[i]);
            if (!valid_candidate) {
                continue;
            }
            int64_t local_pos = static_cast<int64_t>(i - start);
            bool new_cluster = (
                !has_open_cluster ||
                (local_pos - previous_event_pos) > static_cast<int64_t>(gap_ticks)
            );
            if (new_cluster) {
                if (has_open_cluster) {
                    clusters.push_back(open_cluster);
                }
                open_cluster = TickPulseCluster{static_cast<int64_t>(i), cluster_id, 1};
                ++cluster_id;
                has_open_cluster = true;
            } else {
                open_cluster.cluster_size += 1;
            }
            previous_event_pos = local_pos;
        }
        if (has_open_cluster) {
            clusters.push_back(open_cluster);
        }

        start = end;
    }

    auto candidate_indices = py::array_t<int64_t>(clusters.size());
    auto event_cluster_id = py::array_t<int>(clusters.size());
    auto event_cluster_size = py::array_t<int>(clusters.size());
    auto candidate_buf = candidate_indices.request();
    auto cluster_id_buf = event_cluster_id.request();
    auto cluster_size_buf = event_cluster_size.request();
    int64_t* candidate_ptr = static_cast<int64_t*>(candidate_buf.ptr);
    int* cluster_id_ptr = static_cast<int*>(cluster_id_buf.ptr);
    int* cluster_size_ptr = static_cast<int*>(cluster_size_buf.ptr);
    for (size_t i = 0; i < clusters.size(); ++i) {
        candidate_ptr[i] = clusters[i].first_index;
        cluster_id_ptr[i] = clusters[i].cluster_id;
        cluster_size_ptr[i] = clusters[i].cluster_size;
    }

    py::dict result;
    result["future_mid_price"] = future_mid_price;
    result["future_move_ticks"] = future_move_ticks;
    result["expected_direction_code"] = expected_direction_code;
    result["criterion_flow"] = criterion_flow;
    result["criterion_book"] = criterion_book;
    result["criterion_price_resilience"] = criterion_price_resilience;
    result["criterion_volume_burst"] = criterion_volume_burst;
    result["criterion_price_shock"] = criterion_price_shock;
    result["hypothesis_signal"] = hypothesis_signal;
    result["is_correct"] = is_correct;
    result["candidate_indices"] = candidate_indices;
    result["event_cluster_id"] = event_cluster_id;
    result["event_cluster_size"] = event_cluster_size;
    return result;
}

// ==========================================
// 4. PYBIND11 BINDINGS
// ==========================================
PYBIND11_MODULE(_quant_core, m) {
    m.doc() = "Institutional OOP C++ HFT Execution Engine";
    
    // Bind the Hurst Wavelet Estimator as a standalone static function!
    m.def("estimate_hurst", &WaveletHurstEstimator::estimate_hurst, 
          "Calculate Hurst Exponent using Haar Wavelet Transform", 
          py::arg("prices"));

    m.def("compute_tick_rtv_pipeline", &compute_tick_rtv_pipeline,
          "Compute Relative Tick Velocity features, horizon labels, and collapsed events.",
          py::arg("mid_price"), py::arg("tick_size"), py::arg("symbol_id"), py::arg("session_id"),
          py::arg("horizon_ticks"), py::arg("fast_window"), py::arg("slow_window"),
          py::arg("percentile"), py::arg("min_periods"), py::arg("min_fast_move_ticks"),
          py::arg("min_success_ticks"), py::arg("fade"), py::arg("gap_ticks"));

    m.def("compute_tick_pulse_features_core", &compute_tick_pulse_features_core,
          "Compute tick pulse numeric feature columns for the research dashboard.",
          py::arg("last_price"), py::arg("volume"), py::arg("bid_price_1"),
          py::arg("bid_volume_1"), py::arg("ask_price_1"), py::arg("ask_volume_1"),
          py::arg("oi"), py::arg("tick_size"), py::arg("symbol_id"),
          py::arg("session_id"), py::arg("window"), py::arg("min_periods"));

    m.def("compute_tick_heuristic_pipeline", &compute_tick_heuristic_pipeline,
          "Compute non-RTV tick pulse hypothesis labels and collapsed events.",
          py::arg("mid_price"), py::arg("tick_size"), py::arg("flow_imbalance"),
          py::arg("book_imbalance"), py::arg("volume_intensity"),
          py::arg("rolling_mid_move_ticks"), py::arg("price_shock"),
          py::arg("symbol_id"), py::arg("session_id"), py::arg("horizon_ticks"),
          py::arg("hypothesis_code"), py::arg("min_success_ticks"),
          py::arg("flow_sell_max"), py::arg("book_buy_min"), py::arg("book_sell_max"),
          py::arg("breakdown_book_max"), py::arg("volume_burst_min"),
          py::arg("breakdown_volume_burst_min"), py::arg("rolling_mid_up_min"),
          py::arg("rolling_mid_down_max"), py::arg("breakdown_rolling_mid_max"),
          py::arg("price_shock_min"), py::arg("gap_ticks"));

    py::class_<ITCAModel, std::shared_ptr<ITCAModel>>(m, "ITCAModel");
    py::class_<IMarginModel, std::shared_ptr<IMarginModel>>(m, "IMarginModel");

    py::class_<SquareRootTCA, ITCAModel, std::shared_ptr<SquareRootTCA>>(m, "SquareRootTCA")
        .def(py::init<double, double>(), py::arg("bps") = 0.00015, py::arg("gamma") = 0.1);

    py::class_<CryptoOrderBookTCA, ITCAModel, std::shared_ptr<CryptoOrderBookTCA>>(m, "CryptoOrderBookTCA")
        .def(py::init<double, double>(), py::arg("bps") = 0.0005, py::arg("penalty") = 2.0);

    // NEW: Bind the Stochastic TCA Wrapper
    py::class_<StochasticTCAWrapper, ITCAModel, std::shared_ptr<StochasticTCAWrapper>>(m, "StochasticTCAWrapper")
        .def(py::init<double, double, double, int>(), 
             py::arg("lambda") = 1e-4, py::arg("eta") = 0.1, py::arg("gamma") = 2.0, py::arg("T") = 60);

    py::class_<FuturesMargin, IMarginModel, std::shared_ptr<FuturesMargin>>(m, "FuturesMargin")
        .def(py::init<double>(), py::arg("maintenance_req") = 0.05);

    py::class_<EquitiesMargin, IMarginModel, std::shared_ptr<EquitiesMargin>>(m, "EquitiesMargin")
        .def(py::init<double>(), py::arg("maintenance_req") = 0.25);

    py::class_<ExecutionEngine>(m, "ExecutionEngine")
        .def(py::init<std::shared_ptr<ITCAModel>, std::shared_ptr<IMarginModel>, double, double, bool, bool, double>(),
             py::arg("tca_model"), py::arg("margin_model"), py::arg("initial_capital") = 1000000.0,
             py::arg("deadband") = 0.015, py::arg("enforce_price_limits") = false, py::arg("enforce_t1") = false,
             py::arg("fixed_slippage_ticks_per_side") = 0.0)
        // UPDATED BINDING: Accepts volatilities and hursts
        .def("run_simulation", &ExecutionEngine::run_simulation,
             py::arg("asset_ids"), py::arg("prices"), py::arg("target_weights"), 
             py::arg("volumes"), py::arg("volatilities"), py::arg("hursts"),
             py::arg("date_ids") = py::array_t<int>())
        .def("run_simulation_with_costs", &ExecutionEngine::run_simulation_with_costs,
             py::arg("asset_ids"), py::arg("prices"), py::arg("target_weights"),
             py::arg("volumes"), py::arg("volatilities"), py::arg("hursts"),
             py::arg("date_ids"), py::arg("multipliers"), py::arg("fee_types"),
             py::arg("fee_open"), py::arg("fee_close_history"), py::arg("fee_close_today"),
             py::arg("tick_sizes") = py::array_t<double>(), py::arg("integer_lots") = false)
        .def("run_simulation_with_costs_and_returns", &ExecutionEngine::run_simulation_with_costs_and_returns,
             py::arg("asset_ids"), py::arg("prices"), py::arg("target_weights"),
             py::arg("period_returns"), py::arg("volumes"), py::arg("volatilities"),
             py::arg("hursts"), py::arg("time_ids"), py::arg("date_ids"), py::arg("multipliers"),
             py::arg("fee_types"), py::arg("fee_open"), py::arg("fee_close_history"),
             py::arg("fee_close_today"), py::arg("tick_sizes") = py::array_t<double>(),
             py::arg("integer_lots") = false);
}

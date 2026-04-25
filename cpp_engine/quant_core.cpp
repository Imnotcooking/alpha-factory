#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Institutional Backtest Simulator: Applies Slippage and Non-Linear Alpha Decay
py::array_t<double> simulate_execution(py::array_t<double> returns,
                                       double slippage_bps,
                                       double decay_factor) {
  py::buffer_info buf = returns.request();
  double *ptr = static_cast<double *>(buf.ptr);

  // Create the output array
  auto result = py::array_t<double>(buf.size);
  py::buffer_info result_buf = result.request();
  double *result_ptr = static_cast<double *>(result_buf.ptr);

  double slippage_decimal = slippage_bps / 10000.0;

  // O(N) loop running at native C++ speed
  for (size_t i = 0; i < buf.size; i++) {
    double raw_return = ptr[i];

    // Only apply slippage on days the strategy actually trades (non-zero
    // returns)
    double trade_cost = (raw_return != 0.0) ? slippage_decimal : 0.0;

    // Exponential Alpha Decay: The edge degrades as time 'i' moves forward
    // decay_factor scales how fast the edge disappears (e.g., 0.0001)
    double decay_penalty = std::exp(-decay_factor * i);

    // If raw_return is positive (a win), the decay penalty reduces it.
    // If it's negative (a loss), we don't artificially reduce the loss.
    double adjusted_return = raw_return;
    if (raw_return > 0) {
      adjusted_return = raw_return * decay_penalty;
    }

    result_ptr[i] = adjusted_return - trade_cost;
  }

  return result;
}

PYBIND11_MODULE(quant_core, m) {
  m.doc() = "C++ Accelerated Quant Engine for Oxford MCF Portfolio";
  m.def("simulate_execution", &simulate_execution,
        "Applies slippage and alpha decay to a returns array");
}
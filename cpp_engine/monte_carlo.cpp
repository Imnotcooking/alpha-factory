#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// A highly optimized C++ function for heavy backtesting loops
py::array_t<double> compute_slippage_and_decay(py::array_t<double> returns,
                                               double slippage_bps) {
  py::buffer_info buf = returns.request();
  double *ptr = static_cast<double *>(buf.ptr);

  // Create a new NumPy array to return to Python
  auto result = py::array_t<double>(buf.size);
  py::buffer_info result_buf = result.request();
  double *result_ptr = static_cast<double *>(result_buf.ptr);

  // O(N) loop running at native C++ speed
  for (size_t i = 0; i < buf.size; i++) {
    // Example: Apply slippage and a simulated Bayesian haircut decay
    double raw_return = ptr[i];
    double haircut = (i * 0.00001); // Simulated alpha decay over time
    result_ptr[i] = raw_return - (slippage_bps / 10000.0) - haircut;
  }

  return result;
}

// Bind the function to Python
PYBIND11_MODULE(quant_core, m) {
  m.doc() = "C++ Accelerated Quant Core for Streamlit";
  m.def("compute_slippage_and_decay", &compute_slippage_and_decay,
        "Calculates backtest slippage");
}
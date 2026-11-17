// The box cuts off the unconstrained optimum (1, 1).
// On x = 0.8, the constrained optimum is y = x².
export const rosenbrock = {
  initial: [-1.2, 1],
  lower: [-2, -1],
  upper: [0.8, 3],
  evaluate([x, y]) {
    const residual = y - x * x;
    return {
      value: (1 - x) ** 2 + 100 * residual ** 2,
      gradient: [2 * (x - 1) - 400 * x * residual, 200 * residual],
    };
  },
};

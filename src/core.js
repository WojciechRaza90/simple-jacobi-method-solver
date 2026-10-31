export class NonConvergenceError extends Error {
  constructor(iterations, residual) {
    super(`Jacobi method did not converge after ${iterations} iterations (residual ${residual})`);
    this.name = 'NonConvergenceError';
    this.iterations = iterations;
    this.residual = residual;
  }
}

export class SingularMatrixError extends Error {
  constructor(message) {
    super(message);
    this.name = 'SingularMatrixError';
  }
}

/**
 * Solves A x = b using the Jacobi method.
 *
 * The Jacobi iteration is:
 *   x_i^{(k+1)} = (b_i - sum_{j != i} A_ij x_j^{(k)}) / A_ii
 *
 * This is a component-wise update that uses only values from the previous
 * iteration. It is guaranteed to converge for strictly diagonally dominant
 * matrices. For other matrices, convergence is not guaranteed; we detect
 * divergence via the residual norm and throw NonConvergenceError.
 *
 * @param {number[][]} A - square matrix (n x n), must have non-zero diagonal
 * @param {number[]} b - right-hand side vector (length n)
 * @param {object} [options]
 * @param {number} [options.maxIterations=1000]
 * @param {number} [options.tolerance=1e-10]
 * @param {number[]} [options.initialGuess] - starting x, defaults to zeros
 * @returns {number[]} solution vector x
 * @throws {SingularMatrixError} if A is not square, has zero diagonal, or dimensions mismatch
 * @throws {NonConvergenceError} if maxIterations is reached before tolerance
 */
export function jacobi(A, b, options = {}) {
  const { maxIterations = 1000, tolerance = 1e-10, initialGuess } = options;

  if (!Array.isArray(A) || A.length === 0) {
    throw new SingularMatrixError('A must be a non-empty square matrix');
  }
  const n = A.length;
  if (!Array.isArray(b) || b.length !== n) {
    throw new SingularMatrixError('b must have the same length as A');
  }
  for (let i = 0; i < n; i++) {
    if (!Array.isArray(A[i]) || A[i].length !== n) {
      throw new SingularMatrixError('A must be a square matrix');
    }
    if (A[i][i] === 0) {
      throw new SingularMatrixError(`A[${i}][${i}] is zero; Jacobi method requires non-zero diagonal`);
    }
  }

  let x;
  if (initialGuess === undefined) {
    x = new Array(n).fill(0);
  } else {
    if (!Array.isArray(initialGuess) || initialGuess.length !== n) {
      throw new SingularMatrixError('initialGuess must have the same length as A');
    }
    x = initialGuess.slice();
  }

  const xNew = new Array(n);

  for (let iter = 0; iter < maxIterations; iter++) {
    for (let i = 0; i < n; i++) {
      let sum = b[i];
      for (let j = 0; j < n; j++) {
        if (j !== i) {
          sum -= A[i][j] * x[j];
        }
      }
      xNew[i] = sum / A[i][i];
    }

    // compute residual ||A x_new - b||_inf to check convergence
    let maxResidual = 0;
    for (let i = 0; i < n; i++) {
      let rowSum = b[i];
      for (let j = 0; j < n; j++) {
        rowSum -= A[i][j] * xNew[j];
      }
      const abs = Math.abs(rowSum);
      if (abs > maxResidual) {
        maxResidual = abs;
      }
    }

    // swap x and xNew
    for (let i = 0; i < n; i++) {
      x[i] = xNew[i];
    }

    if (maxResidual < tolerance) {
      return x;
    }
  }

  // compute final residual for the error message
  let maxResidual = 0;
  for (let i = 0; i < n; i++) {
    let rowSum = b[i];
    for (let j = 0; j < n; j++) {
      rowSum -= A[i][j] * x[j];
    }
    const abs = Math.abs(rowSum);
    if (abs > maxResidual) {
      maxResidual = abs;
    }
  }

  throw new NonConvergenceError(maxIterations, maxResidual);
}

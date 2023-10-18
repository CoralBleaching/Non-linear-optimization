#pragma once

#include <functional>
#include <iostream>
#include <limits>
#include <cmath>

#include "../linear_algebra.hpp"


/*
TODO:
- refactor names
- add "termination parameter" instead of error
*/
namespace alg 
{
    using std::log;                     

    /* 
    There should be some performance hit for using std::function instead of "auto" returning capturing lambdas, but
    it allows IntelliSense to recognize all operations, and this really helps catching errors in complex formulas.
    */
    using RealFunction = std::function<double(Vector)>;                // A function that takes a vector argument and returns a scalar.
    using VectorFunction = std::function<Vector(Vector)>;              // A vector function; takes a vector argument and returns another vector. Useful for gradients.
    using MatrixFunction = std::function<Matrix(Vector)>;              // A matrix function; takes a vector argument and returns a matrix. Useful for Jacobians and Hessians.
    using PhiFunction = std::function<double(Vector, double, Vector)>; // A specific type of function recurrent in optimization routines.

    constexpr double fInf = std::numeric_limits<double>::infinity();
    const std::vector<RealFunction> no_restrictions{};

    size_t nTerminationCutoff = 100'000;
    void setTerminationCutoff(size_t cutoff) { nTerminationCutoff = cutoff; }

    std::vector<std::pair<size_t, size_t>> countingStack;
    void pushCount(size_t cap = nTerminationCutoff) { countingStack.push_back({ cap,0u }); }
    void popCount() { if (!countingStack.empty()) countingStack.pop_back(); }
    bool checkCount()
    {
        if (!countingStack.empty()) {
            return ++countingStack.back().second > countingStack.back().first;
        }
        else
            return false;
    }

    /*
        Takes a real function func of n variables and returns a vector_function that evaluates the gradient of func
        at any given x = { x1, x2, ..., xn}. A central-difference formula is used.
        Parameters:
                    func - a function that takes a vector of doubles and returns a double
                    n    - the size of the argument to f (i.e. number of variables in the function)
                    e    - the precision of the approximation to the derivatives of f
        Returns:
                    function - a function that takes a vector of doubles and returns another vector of doubles.
                                    Parameters:
                                                x - the point (vector of doubles) at which to evaluate the gradient of f.
                                    Returns:
                                                g - a vector of doubles representing the gradient of f at x.
    */
    VectorFunction gradient(RealFunction func, int n, double e = 1e-6)
    {
        Matrix E = e * I(n);
        return [func, n, e, E](Vector x) {
            Vector g(n);
            for (int i = 0; i < n; i++)
            {
                g[i] = (func(x + E[i]) - func(x - E[i])) / (2 * e);
            }
            return g;
        };
    }

    /*
        Takes a vector function G of n variables and returns a Matrix function that evaluates the Jacobian
        matrix of that function at any given x = { x1, x2, ..., xn}. A central-difference formula is used.
        Parameters:
                    G - a function that takes a Vector of doubles and returns another Vector of doubles.
                    n - the size of the argument to G (i.e. number of variables in the function)
                    e - the precision of the approximation to the derivatives of G
        Returns:
                    function - a function that takes a Vector of doubles and returns a Matrix.
                                    Parameters:
                                                x - the point (vector of doubles) at which to evaluate the Jacobian of G.
                                    Returns:
                                                H - a matrix of doubles representing the Jacobian of G at x.
    */
    MatrixFunction jacobian(VectorFunction G, int n, double e = 1e-4)
    {
        return [G, n, e](Vector x) {
            Matrix H(n, n);
            Matrix M = e * I(n);
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < n; j++)
                    H(i, j) = (G(x + M[j]) - G(x - M[j]))[i] / (4 * e) + (G(x + M[i]) - G(x - M[i]))[j] / (4 * e);
            H = 0.5 * (H + t(H));
            return H;
        };
    }

    /*
        Algorithm for the search of step size to be utilized in optimizing routines. 'phi' represents the
        mathematical function:
            phi(x,t,d) = f(x + td).
        The 'x' and 'd' arguments are necessary as we must update phi at every step of the superjacent optimizing routine.
    */
    double step_golden_section(PhiFunction phi, Vector x, Vector d, double error = 1e-6, double rho = 1)
    {
        double theta1 = (3 - sqrt(5)) / 2.;
        double theta2 = 1 - theta1;
        // stage 1 - obtaining the interval [a, b]
        double a = 0;
        double s = rho;
        double b = 2 * rho;
        pushCount();
        while (phi(x, b, d) < phi(x, s, d))
        {
            a = s;
            s = b;
            b *= 2;
            if (checkCount()) { std::cout << "EXCEEDED COUNT.\n"; break; }
        }
        // stage 2 - obtaining t_bar in [a, b]
        double u = a + theta1 * (b - a);
        double v = a + theta2 * (b - a);
        while (b - a > error)
        {
            if (phi(x, u, d) < phi(x, v, d))
            {
                b = v;
                v = u;
                u = a + theta1 * (b - a);
            }
            else
            {
                a = u;
                u = v;
                v = a + theta2 * (b - a);
            }
            if (checkCount()) { std::cout << "EXCEEDED COUNT.\n"; break; }
        }
        popCount();
        return u / 2. + v / 2.;
    }

    /*
        Shorthand to calculate the one-dimensional derivatives of so-called "phi functions" as defined above
        by using the 'grad' function previously. It actually returns a function that evaluates the derivative
        at point 't'.
    */
    std::function<double(double)> D(PhiFunction phi, Vector x, Vector d, double error = 1e-6)
    {
        return [phi, x, d, error](double t) {
            return (phi(x, t + error, d) - phi(x, t - error, d)) / (2 * error);
        };
    }


    /*
        Shorthand to calculate the two-dimensional derivatives of so-called "phi functions" as defined above. 
        It actually returns a function that evaluates the second derivative at point 't'.
    */
    std::function<double(double)> D2(PhiFunction phi, Vector x, Vector d, double error = 1e-6)
    {
        return [phi, x, d, error](double t) {
            return (phi(x, t + 2 * error, d) - 2 * phi(x, t, d) + phi(x, t - 2 * error, d)) / (4 * error * error);
        };
    }

    /*
        Algorithm for the search of step size to be utilized in optimizing routines. Numerically approximates
        the first and second derivatives of the function 'phi(t)'.
    */
    double step_newton_method(PhiFunction phi, Vector x, Vector d, double t_initial, double error = 1e-6)
    {
        double t_previous = -fInf;
        double t = t_initial;
        auto d_phi_dt = D(phi, x, d, error);
        auto d2_phi_dt2 = D2(phi, x, d);
        pushCount();
        while (std::fabs(t - t_previous) > error)
        {
            t_previous = t;
            if (d_phi_dt(t) == 0) break;
            if (d2_phi_dt2(t) == 0) break;
            if (checkCount()) { std::cout << "EXCEEDED COUNT.\n"; break; }
            t = t_previous - d_phi_dt(t_previous) / d2_phi_dt2(t_previous);
        }
        popCount();
        return t;
    }

    /*
        Another alternative algorithm for the search of step size to be utilized in optimizing routines.
        For clarity, this time we won't transform the original function 'f' into an one-dimensional 'phi'
        function.
    */
    double step_armijo_criterion(RealFunction func, VectorFunction grad, Vector x, Vector direction, double eta = 0.5)
    {
        double t = 1;
        pushCount();
        while (true)
        {
            double linear_delta = eta * t * grad(x) * direction;
            double step = func(x + t * direction); 
            if (std::isnan(step)) break;//throw std::exception("Armijo step search: step size is NaN.");
            if (step <= func(x) + linear_delta)
                break;
            t *= 0.8;
            if (checkCount()) { std::cout << "EXCEEDED COUNT.\n"; break; }
        }
        popCount();
        return t;
    }

    // Short exception type to specify user mistakes in data entry.
    class InvalidOptionException : public std::exception
    {
        const char* what()  const noexcept override
        {
            return "Invalid option selected for parameter.";
        }
    };

    /*
        Returns true if any of the given restrictions is violated, i.e. any of them returns a positive number.
        The restrictions are a set of functions given by the user that will be evaluated at x. The convention
        adopted here is that restrictions should be of the negative type: their value should always be
        negative within the feasible region.
    */
    bool check_inequality_restrictions(std::vector<RealFunction> restrictions, Vector x)
    {
        for (auto& restriction : restrictions)
            if (restriction(x) > 0) return false;
        return true;
    }

    /*
        Approximates the minimum value of a function to within a given precision via the Newton method.
        Parameters such as error and precision have default values for ease of use. The 'sigma'
        parameter ensures global convergence and has default value of 1, but can be modified. 'sigma = 0' will
        imply a non-modified Newton method, with loss of global convergence.
        Parameters:
            Required:
                func         - a function representing f(x1, ..., xn) to be minimized. The type
                            'RealFunction' was described above.
                n            - the size of the argument to func, i.e. the number of variables of f(x1, ..., xn).
                restrictions - functions that describe the boundaries of the region that the algorith must search.
                x_initial    - a vector representing the starting point x^0 of the procedure.
                x_final      - reference to a vector representing the minimum point found.
                trajectory   - reference to a "list" of points that records every intermediate step.
            Optional:
                linear_search_method
                            - the method for calculating the step size. Options are: "newton",
                                    "golden-section" and "armijo". Default: "newton".
                sigma        - modifies the Newton method to ensure convergence (by ensuring the Hessian
                            matrix is positive definite). Default: 1.
                error        - the bound on the error on the approximation to the minimum. Default: 1e-6.
                precision    - the bound on the error on the approximation to derivatives used in subroutines. Default: 1e-6
                rho          - used for step size determination within the golden-section method. Default: 1.
                eta          - used for step size determination within the Armijo method. It represents the minimum
                            ratio of reduction obtained in the linear model of the objective function. Default: 0.5.
        Usage:
            The user should define an objective function as a C++ function that takes a vector argument and
            returns a double (scalar). They should likewise define a list of such functions to represent
            the problem's restrictions (adopted convention is that restrictions should be negative). The user
            should also define a vector representing the starting point and declare a vector that will hold
            the finishing point and a matrix that will hold all the visited points. These variables should all
            be passed to the function. Results will be stored in the variables 'x_final' and 'trajectory' that
            were passed as arguments.
    */
    void minimize_newton_method(RealFunction func, int n, std::vector<RealFunction> restrictions,
        Vector x_initial, Vector& x_final, Matrix& trajectory, std::string linear_search_method = "newton",
        double sigma = 0, double error = 1e-6, double precision = 1e-6, double rho = 1, double eta = 0.5)
    {
        VectorFunction gradf = gradient(func, n, precision);    // gradient function of f(x)
        MatrixFunction hessf = jacobian(gradf, n, precision);  // hessian function of f(x)

        Vector direction = -gradf(x_initial);
        x_final = x_initial;
        trajectory.insertRow(x_final);
        // linearization of f(x) in order to determine the step size in the direction 'd'.
        PhiFunction phi = [func](Vector x, double t, Vector d) {
            return func(x + t * d);
        };

        // main loop
        pushCount(10);
        while (gradf(x_final).norm() > error)
        {
            // step size calculation
            double t_k;
            if (linear_search_method == "armijo")
                t_k = step_armijo_criterion(func, gradf, x_final, direction, eta);
            else if (linear_search_method == "newton")
            {
                t_k = step_newton_method(phi, x_final, direction, 1, precision);
            }
            else if (linear_search_method == "golden-section")
                t_k = step_golden_section(phi, x_final, direction, error, rho);
            else
                throw InvalidOptionException();
            if (std::isnan(t_k)) break;

            // updating direction
            Matrix M_k = Matrix();
            // calculating M factor: inverse of the modified hessian (by the sigma parameter)
            if (sigma > 0) M_k = (hessf(x_final) + sigma * I(n)).inv();
            else           M_k = hessf(x_final).inv();
            // direction calculation under Newton's method
            direction = -M_k * gradf(x_final);
            while (restrictions.size() > 0 && check_inequality_restrictions(restrictions, x_final + t_k * direction) == false)
                t_k *= 0.9; // if g_i(x_k) >= 0 for some i, let's try to "walk back" a little on direction d to the feasible region
            if ((t_k * direction).norm() <= error) // we walked back until we're too close to x_{k-1}
            {
                popCount();
                return;
            }

            // updating x
            x_final += t_k * direction;

            // recording the current step
            trajectory.insertRow(x_final);

            // safety measure to avoid division by zero (stationary point prematurely reached).
            if (trajectory.nrows() > 1 && *(trajectory.end() - 1) == *(trajectory.end() - 2))
            {
                std::cout << "Warning: Premature break." << std::endl;
                break;
            }
            // if the algorithm is taking too long, occasionally check to see if it's making progress
            if (trajectory.nrows() % (size_t)1e3 == 0) std::cout << trajectory.nrows() << ": " << *(trajectory.end() - 1) << std::endl;

            if (checkCount()) { std::cout << "EXCEEDED COUNT.\n"; break; }
        }
        popCount();
    }


    /*
        Approximates the minimum value of a function to within a given precision via the Conjugate gradient method.
        Parameters such as error and precision have default values for ease of use. The conjugate matrix of choice
        is the hessian matrix. This method also allows a cheaper fourth method of calculating the step size.
        Parameters:
            Required:
                func         - a function representing f(x1, ..., xn) to be minimized. The type
                            'real_function' was described above.
                n            - the size of the argument to func, i.e. the number of variables of f(x1, ..., xn).
                restrictions - functions that describe the boundaries of the region that the algorith must search.
                x_initial    - a vector representing the starting point x^0 of the procedure.
                x_final      - reference to a vector representing the minimum point found.
                trajectory   - reference to a "list" of points that records every intermediate step.
            Optional:
                linear_search_method
                            - the method for calculating the step size. Options are: "newton",
                                    "golden-section", "armijo", and "quadratic". Default: "newton".
                error        - the bound on the error on the approximation to the minimum. Default: 1e-6.
                precision    - the bound on the error on the approximation to derivatives used in subroutines. Default: 1e-6
                rho          - used for step size determination within the golden-section method. Default: 1.
                eta          - used for step size determination within the Armijo method. It represents the minimum
                            ratio of reduction obtained in the linear model of the objective function. Default: 0.5.
        Usage:
            The user should define an objective function as a C++ function that takes a vector argument and
            returns a double (scalar). They should likewise define a list of such functions to represent
            the problem's restrictions (adopted convention is that restrictions should be negative). The user
            should also define a vector representing the starting point and declare both a vector that will hold
            the finishing point and a matrix that will hold all the visited points. These variables should all
            be passed to the function. Results will be stored in the variables 'x_final' and 'trajectory' that
            were passed as arguments.
    */
    void minimize_conjugate_gradient(RealFunction func, int n, std::vector<RealFunction> restrictions,
        Vector x_initial, Vector& x_final, Matrix& trajectory, std::string linear_search_method = "newton",
        double error = 1e-6, double precision = 1e-6, double rho = 1, double eta = 0.5)
    {
        VectorFunction gradf = gradient(func, n, precision);    // gradient function of f(x)
        MatrixFunction hessf = jacobian(gradf, n, precision);  // hessian function of f(x)

        Vector direction = -gradf(x_initial);
        x_final = x_initial;
        trajectory.insertRow(x_final);
        // linearization of f(x) in order to determined the step size in the direction 'd'.
        PhiFunction phi = [func](Vector x, double t, Vector d) {
            return func(x + t * d);
        };

        // main loop
        pushCount(10);
        while (gradf(x_final).norm() > error)
        {
            // step size calculation
            double t_k;
            if (linear_search_method == "armijo")
                t_k = step_armijo_criterion(func, gradf, x_final, direction, eta);
            else if (linear_search_method == "newton")
            {
                t_k = step_newton_method(phi, x_final, direction, 1, precision);
            }
            else if (linear_search_method == "golden-section")
                t_k = step_golden_section(phi, x_final, direction, error, rho);
            else if (linear_search_method == "quadratic")
                t_k = (-gradf(x_final) * direction) / ((direction * hessf(x_final)) * direction);
            else
                throw InvalidOptionException();
            if (std::isnan(t_k)) break;
            // if g_i(x_k) >= 0 for some i, let's try to "walk back" a little on direction d to the feasible region
            while (restrictions.size() > 0 && check_inequality_restrictions(restrictions, x_final + t_k * direction) == false)
                t_k *= 0.9;
            if (norm(t_k * direction) <= error) // we walked back until we're too close to x_{k-1}
            {
                popCount();
                return;
            }

            // updating x
            Vector x_prev = x_final;
            x_final += +t_k * direction;

            // updating direction
            double beta_k = (direction * hessf(x_final) * gradf(x_final)) / (direction * hessf(x_final) * direction);
            direction = -gradf(x_final) + beta_k * direction;

            // recording the current step
            trajectory.insertRow(x_final);

            // safety measure to avoid division by zero (stationary point prematurely reached).
            if (trajectory.nrows() > 1 && *(trajectory.end() - 1) == *(trajectory.end() - 2))
            {
                cout << "Warning: Premature break." << endl;
                break;
            }
            // if the algorithm is taking too long, occasionally check to see if it's making progress
            if (trajectory.nrows() % (size_t)1e3 == 0) cout << trajectory.nrows() << ": " << *(trajectory.end() - 1) << endl;

            if (checkCount()) { std::cout << "EXCEEDED COUNT.\n"; break; }
        }
        popCount();
    }


    /*
        Approximates the minimum value of a function to within a given precision via the Quasi-Newton BFGS method.
        Parameters such as error and precision have default values for ease of use. The 'beta' parameter
        alters the initial estimation of the hessian.
        Parameters:
            Required:
                func         - a function representing f(x1, ..., xn) to be minimized. The type
                            'real_function' was described above.
                n            - the size of the argument to func, i.e. the number of variables of f(x1, ..., xn).
                restrictions - functions that describe the boundaries of the region that the algorith must search.
                x_initial    - a vector representing the starting point x^0 of the procedure.
                x_final      - reference to a vector representing the minimum point found.
                trajectory   - reference to a "list" of points that records every intermediate step.
            Optional:
                linear_search_method
                            - the method for calculating the step size. Options are: "newton",
                                    "golden-section" and "armijo". Default: "newton".
                beta         - modifies the initial estimation of the hessian matrix (identity matrix) by
                            a scalar factor. Default: 1.
                error        - the bound on the error on the approximation to the minimum. Default: 1e-6.
                precision    - the bound on the error on the approximation to derivatives used in subroutines. Default: 1e-6
                rho          - used for step size determination within the golden-section method. Default: 1.
                eta          - used for step size determination within the Armijo method. It represents the minimum
                            ratio of reduction obtained in the linear model of the objective function. Default: 0.5.
        Usage:
            The user should define an objective function as a C++ function that takes a vector argument and
            returns a double (scalar). They should likewise define a list of such functions to represent
            the problem's restrictions (adopted convention is that restrictions should be negative). The user
            should also define a vector representing the starting point and declare a vector that will hold
            the finishing point and a matrix that will hold all the visited points. These variables should all
            be passed to the function. Results will be stored in the variables 'x_final' and 'trajectory' that
            were passed as arguments.
    */
    void minimize_quasi_newton_method(RealFunction func, int n, std::vector<RealFunction> restrictions,
        Vector x_initial, Vector& x_final, Matrix& trajectory, std::string linear_search_method = "newton",
        double beta = 1, double error = 1e-6, double precision = 1e-6, double rho = 1, double eta = 0.5)
    {
        VectorFunction gradf = gradient(func, n, precision);    // gradient function of f(x)
        MatrixFunction hessf = jacobian(gradf, n, precision);  // hessian function of f(x)

        Vector direction = -gradf(x_initial);
        x_final = x_initial;
        trajectory.insertRow(x_final);
        // linearization of f(x) in order to determined the step size in the direction 'd'.
        PhiFunction phi = [func](Vector x, double t, Vector d) {
            return func(x + t * d);
        };
        // initial estimation of the hessian
        Matrix H = (beta)*I(n);

        // main loop
        pushCount();
        while (gradf(x_final).norm() > error)
        {
            // determining step size
            double t_k;
            if (linear_search_method == "armijo")
                t_k = step_armijo_criterion(func, gradf, x_final, direction, eta);
            else if (linear_search_method == "newton")
            {
                t_k = step_newton_method(phi, x_final, direction, 1, precision);
            }
            else if (linear_search_method == "golden-section")
                t_k = step_golden_section(phi, x_final, direction, error, rho);
            else
                throw InvalidOptionException();
            if (std::isnan(t_k)) break;

            // updating direction
            direction = -H * gradf(x_final);
            while (restrictions.size() > 0 && check_inequality_restrictions(restrictions, x_final + t_k * direction) == false)
                t_k *= 0.9; // if g_i(x_k) >= 0 for some i, let's try to "walk back" a little on direction d to the feasible region
            if (norm(t_k * direction) <= error) // we walked back until we're too close to x_{k-1}
            {
                popCount();
                return;
            }

            // updating x
            Vector p = t_k * direction;
            x_final += p;
            Vector q = gradf(x_final) - gradf(x_final - p);
            // updating the hessian (BFGS formula)
            H += (1. + (q * H * q) / (p * q)) * ((p.col() * p.row()) / (p * q)) - ((p.col() * q.row()) * H + H * (q.col() * p.row())) / (p * q);

            // recording the current step
            trajectory.insertRow(x_final);
            // safety measure to avoid division by zero (stationary point prematurely reached).
            if (trajectory.nrows() > 1 && *(trajectory.end() - 1) == *(trajectory.end() - 2))
            {
                cout << "Warning: Premature break." << endl;
                break;
            }
            // if the algorithm is taking too long, occasionally check to see if it's making progress
            if (trajectory.nrows() % (size_t)1e3 == 0) cout << trajectory.nrows() << ": " << *(trajectory.end() - 1) << endl;

            if (checkCount()) { std::cout << "EXCEEDED COUNT.\n"; break; }
        }
        popCount();
    }

    /*
        Returns a function that evaluates the value of a mathematical "barrier" function generated from the
        set of given restriction functions. To be used in generating subproblems modified by adding a multiple
        of this barrier function to the originial objective function in the "barrier method" of optimization.
        Parameters:
            restrictions - (vector/list of) functions that describe the boundaries of the region that the algorith must search.
            type         - the type of calculation to be made upon the restrictions in order to produce a
                        barrier function that goes to infinity as the number of steps increase.
    */
    RealFunction generate_barrier_function(std::vector<RealFunction> restrictions, std::string type = "log")
    {
        if (type == "log")
            return [restrictions](Vector x) {
            double sum = 0;
            for (auto& restriction : restrictions)
            {
                sum += log(-restriction(x));
            }
            return -sum;
        };
        else if (type == "inverse")
            return [restrictions](Vector x) {
            double sum = 0;
            for (auto& restriction : restrictions)
            {
                sum += -1 / restriction(x);
            }
            return -sum;
        };
        else throw InvalidOptionException();
    }


    /*
        Approximates the minimum value of a function in a region defined by user given restrictions to within
        a given precision via the barrier method, which solves a sequence of subproblems to approximate the
        solution to the primal problem given. Parameters such as error and precision have default values for
        ease of use.

        Parameters:
            Required:
                func         - a function representing f(x1, ..., xn) to be minimized. The type
                            'real_function' was described above.
                n            - the size of the argument to func, i.e. the number of variables of f(x1, ..., xn).
                restrictions - functions that describe the boundaries of the region that the algorith must search.
                x_initial    - a vector representing the starting point x^0 of the procedure.
                x_final      - reference to a vector representing the minimum point found.
                trajectory   - reference to a "list" of points that records every intermediate step.
            Optional:
                minimization_method
                            - the underlying method for optimizing (minimizing) the subproblems. Options:
                            "newton-method", "conjugate-gradient", and "quasi-newton". Default: "conjugate-gradient".
                linear_search_method
                            - the method for calculating the step size. Options are: "newton",
                                    "golden-section" and "armijo". Default: "newton".
                barrier_type - the type of calculation to be made upon the restrictions in order to produce a
                            barrier function that goes to infinity as the number of steps increase. Default: "log".
                mu           - initial value of the "penalty" parameter. Default: 10.0.
                beta         - Scaling factor to the mu parameter. Default: 0.1.
                error        - the bound on the error on the approximation to the minimum. Default: 1e-6.
                precision    - the bound on the error on the approximation to derivatives used in subroutines. Default: 1e-6
                rho          - used for step size determination within the golden-section method. Default: 1.
                eta          - used for step size determination within the Armijo method. It represents the minimum
                            ratio of reduction obtained in the linear model of the objective function. Default: 0.5.
                sigma        - has different meanings depending on the subjacent unrestricted optimization method
                            chosen. "newton": level of approximation to the gradient method (sigma > 0 NOT recommended
                            for restricted optimization). "quasi-newton": multiplier to the initial estimate of
                            the hessian of f(x) (the objective function).

        Usage:
            The user should define an objective function as a C++ function that takes a vector argument and
            returns a double (scalar). They should likewise define a list of such functions to represent
            the problem's restrictions (adopted convention is that restrictions should be negative). The user
            should also define a vector representing the starting point and declare a vector that will hold
            the finishing point and a matrix that will hold all the visited points. These variables should all
            be passed to the function. Results will be stored in the variables 'x_final' and 'trajectory' that
            were passed as arguments.

            Example of declaring lists of restrictions:
                // For f(x = {x1, ..., xn}) we can define a restrictions of the form x1 > -2 and x1 + x3 < 0.
                // In our current conventions, such restrictions would be defined as follows:
                std::vector<real_function> restrictions = {
                    [](Vector x) { return -x[0] - 2; },    // x1 + 2 > 0
                    [](Vector x) { return x[0] + x[2]; }   // x1 + x3 < 0
                };

        Effects:
            The function prints some information to the console every time it finishes a subproblem. The
            information is as follows:
                "{current step}: mu: {value of mu}, B(x): { value of barrier at final x }, mu*B(x): { value of mu*B(X) },"
    */
    void minimize_barrier_method(RealFunction func, int n, std::vector<RealFunction> restrictions, Vector x_initial, Vector& x_final, double& f_final,
        Matrix& trajectory, std::string minimization_method = "conjugate-gradient", std::string linear_search_method = "newton",
        std::string barrier_type = "log", double mu = 10, double beta = 0.1, double error = 1e-6,
        double precision = 1e-6, double rho = 1, double eta = 0.5, double sigma = 0)
    {
        if (!check_inequality_restrictions(restrictions, x_initial))
        {
            cout << "Error: infeasible starting point." << endl << endl;
            return;
        }
        Vector x_prev = x_initial;
        RealFunction barrier_function = generate_barrier_function(restrictions, barrier_type);
        pushCount(10);
        while (true)
        {
            RealFunction lagrangian = [func, mu, barrier_function](Vector x) { return func(x) + mu * barrier_function(x); };
            if (minimization_method == "newton-method")
                minimize_newton_method(lagrangian, n, restrictions, x_prev, x_final, trajectory,
                    linear_search_method, sigma, error, precision, rho, eta);
            else if (minimization_method == "conjugate-gradient")
                minimize_conjugate_gradient(lagrangian, n, restrictions, x_prev, x_final, trajectory,
                    linear_search_method, error, precision, rho, eta);
            else if (minimization_method == "quasi-newton")
                minimize_quasi_newton_method(lagrangian, n, restrictions, x_initial, x_final, trajectory,
                    linear_search_method, sigma, error, precision, rho, eta);
            else throw InvalidOptionException();
            cout << trajectory.nrows() << ": " << "mu: " << mu << ", B(x): " << barrier_function(x_final) << ", mu*B(x): " << mu * barrier_function(x_final);
            cout << ", g(x) <= 0: " << ((check_inequality_restrictions(restrictions, x_final)) ? "true" : "false") << endl;
            if (std::abs(mu * barrier_function(x_final)) < error)
                break;
            mu *= beta;
            x_prev = x_final;
            if (checkCount()) { std::cout << "EXCEEDED COUNT.\n"; break; }
        }
        f_final = func(x_final);
        popCount();
    }


    /*
        Returns a function that evaluates the value of a mathematical "penalty" function generated from the
        set of given restriction functions. To be used in generating subproblems modified by adding a multiple
        of this penalty function to the originial objective function in the "penalty method" of optimization.
        Parameters:
            inequality_restrictions - (vector/list of) functions that define boundaries for the value of the
                                    desired optimum.
            equality_restrictions   - (vector/list of) functions represent equality constraints that the
                                    desired optimum must satisfy.
                                    Example:
                                        // For f(x = {x1, ..., xn}) we can define a restriction of the form
                                        // x1 = -3. In our current conventions, such a restriction would be
                                        // defined as follows:
                                        auto restriction = [](Vector x) { return x[0] + 3; };
            power (Optional)        - the exponential factor of the penalty function. Default: 2.

        Note: General form of a penalty function:
        \alpha(x) = \sum_{i = 1}^m \phi(g_i(x)) + \sum_{i = 1}^l \psi (h_i(x))
        where
        \phi(y) = 0 if y <= 0 and \phi(y) > 0 if y > 0
        \psi(y) = 0 if y = 0  and \psi(y) > 0 if y != 0
        */
    RealFunction generate_penalty_function(std::vector<RealFunction> inequality_restrictions, std::vector<RealFunction> equality_restrictions, unsigned int power = 2)
    {
        std::function<double(double)> phi = [power](double y) { return std::pow(std::max(0., y), power); };
        std::function<double(double)> psi;
        if (power % 2)
            psi = [power](double y) { return std::pow(std::abs(y), power); };
        else
            psi = [power](double y) { return std::pow(y, power); };
        return [phi, psi, equality_restrictions, inequality_restrictions](Vector x) {
            double sum = 0;
            for (auto& restriction : inequality_restrictions)
                sum += phi(restriction(x));
            for (auto& restriction : equality_restrictions)
                sum += psi(restriction(x));
            return sum;
        };
    }

    /*
        Approximates the minimum value of a function in a region defined by user given restrictions to within
        a given precision via the penalty method, which solves a sequence of subproblems to approximate the
        solution to the primal problem given. Parameters such as error and precision have default values for
        ease of use.

        Parameters:
            Required:
                func         - a function representing f(x1, ..., xn) to be minimized. The type
                            'real_function' was described above.
                n            - the size of the argument to func, i.e. the number of variables of f(x1, ..., xn).
                inequality_restrictions
                            - functions that define boundaries for the value of the desired optimum.
                equality_restrictions
                            - functions represent equality constraints that the desired optimum must satisfy.
                x_initial    - a vector representing the starting point x^0 of the procedure.
                x_final      - reference to a vector representing the minimum point found.
                trajectory   - reference to a "list" of points that records every intermediate step.
            Optional:
                minimization_method
                            - the underlying method for optimizing (minimizing) the subproblems. Options:
                            "newton-method", "conjugate-gradient", and "quasi-newton". Default: "conjugate-gradient".
                linear_search_method
                            - the method for calculating the step size. Options are: "newton",
                                    "golden-section" and "armijo". Default: "newton".
                barrier_type - the type of calculation to be made upon the restrictions in order to produce a
                            barrier function that goes to infinity as the number of steps increase. Default: "log".
                mu           - initial value of the "penalty" parameter. Default: 10.0.
                beta         - Scaling factor to the mu parameter. Default: 0.1.
                error        - the bound on the error on the approximation to the minimum. Default: 1e-6.
                precision    - the bound on the error on the approximation to derivatives used in subroutines. Default: 1e-6
                rho          - used for step size determination within the golden-section method. Default: 1.
                eta          - used for step size determination within the Armijo method. It represents the minimum
                            ratio of reduction obtained in the linear model of the objective function. Default: 0.5.
                sigma        - has different meanings depending on the subjacent unrestricted optimization method
                            chosen. "newton": level of approximation to the gradient method (sigma > 0 NOT recommended
                            for restricted optimization). "quasi-newton": multiplier to the initial estimate of
                            the hessian of f(x) (the objective function).

        Usage:
            The user should define an objective function as a C++ function that takes a vector argument and
            returns a double (scalar). They should likewise define a list of such functions to represent
            the problem's restrictions (adopted convention is that restrictions should be negative). The user
            should also define a vector representing the starting point and declare a vector that will hold
            the finishing point and a matrix that will hold all the visited points. These variables should all
            be passed to the function. Results will be stored in the variables 'x_final' and 'trajectory' that
            were passed as arguments.

            Example of declaring lists of restrictions:
                // For f(x = {x1, ..., xn}) we can define a list of inequality restrictions of the form
                // x1 > -2 and x1 + x3 < 0. In our current conventions, such restrictions would be defined
                // as follows:
                std::vector<real_function> inequality_restrictions = {
                    [](Vector x) { return -x[0] - 2; },    // x1 + 2 > 0
                    [](Vector x) { return x[0] + x[2]; }   // x1 + x3 < 0
                };
                // Equality restrictions such as x1 = 4 and x2 + x3 = 0 would be defined as follows:
                std::vector<real_function> equality_restrictions = {
                    [](Vector x) { return x[0] - 4; },
                    [](Vector x) { return x[1] + x[3]; }
                };

        Effects:
            The function prints some information to the console every time it finishes a subproblem. The
            information is as follows:
                "{current step}: mu: {value of mu}, a(x): { value of penalty at final x }, mu*a(x): { value of mu*a(X) },"
    */
    void minimize_penalty_method(RealFunction func, int n, std::vector<RealFunction> inequality_restrictions, std::vector<RealFunction> equality_restrictions, Vector x_initial,
        Vector& x_final, double& f_final, Matrix& trajectory, std::string minimization_method = "conjugate-gradient", std::string linear_search_method = "newton",
        unsigned int power = 2, double mu = 0.1, double beta = 10, double error = 1e-6, double precision = 1e-6, double rho = 1, double eta = 0.5, double sigma = 0)
    {
        auto dont_check_restrictions = std::vector<RealFunction>();
        Vector x_prev = x_initial;
        RealFunction penalty_function = generate_penalty_function(inequality_restrictions, equality_restrictions, power);
        pushCount(10);
        while (true)
        {
            RealFunction lagrangian = [func, mu, penalty_function](Vector x) { return func(x) + mu * penalty_function(x); };
            if (minimization_method == "newton-method")
                minimize_newton_method(lagrangian, n, dont_check_restrictions, x_prev, x_final, trajectory,
                    linear_search_method, sigma, error, precision, rho, eta);
            else if (minimization_method == "conjugate-gradient")
                minimize_conjugate_gradient(lagrangian, n, dont_check_restrictions, x_prev, x_final, trajectory,
                    linear_search_method, error, precision, rho, eta);
            else if (minimization_method == "quasi-newton")
                minimize_quasi_newton_method(lagrangian, n, dont_check_restrictions, x_initial, x_final, trajectory,
                    linear_search_method, sigma, error, precision, rho, eta);
            else throw InvalidOptionException();
            cout << trajectory.nrows() << ": " << "mu: " << mu << ", a(x): " << penalty_function(x_final) << ", mu*a(x): " << mu * penalty_function(x_final) << endl;
            if (std::abs(mu * penalty_function(x_final)) < error)
                break;
            mu *= beta;
            x_prev = x_final;

            if (checkCount()) { std::cout << "EXCEEDED COUNT.\n"; break; }
        }
        f_final = func(x_final);
        popCount();
    }
}
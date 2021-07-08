#include <functional>
#include <limits>
#include "linear_algebra.hh"

#include <iostream>
using std::cout;
using std::endl;

/*
TODO:
- refactor names
- add "termination parameter" instead of error
*/

using std::vector;
using std::function;
using std::log;
using std::string;
using std::exception;

typedef vector<double> vecd;

typedef function<double(vecd)> real_function;
typedef function<vecd(vecd)> vector_function;
typedef function<Matrix<double>(vecd)> matrix_function;
typedef function<double(vecd, double, vecd)> phi_function;

double inf = std::numeric_limits<double>::infinity();
vector<real_function> no_restrictions = vector<real_function>();

vector_function grad(real_function func, int n, double e = 1e-6)
{
    vector<real_function> derivs(n);
    for (int i = 0; i < n; i++)
    {
        derivs[i] = [func, i, e, n](vecd x) {
            Matrix<double> M = e * I<double>(n);
            return (func(x + M[i]) - func(x - M[i])) / (2 * e);
        };
    }
    return [n, derivs](vecd x) {
        vecd g(n);
        for (int i = 0; i < n; i++)
        {
            g[i] = derivs[i](x);
        }
        return g;
    };
}

matrix_function hess(vector_function G, int n, double e = 1e-4)
{
    return [G, n, e](vecd x) {
        Matrix<double> H(n, n);
        Matrix<double> M = e * I<double>(n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                H[i][j] = (G(x + M[j]) - G(x - M[j]))[i]  / (4 * e) + (G(x + M[i]) - G(x - M[i]))[j] / (4 * e);
        H = 0.5 * (H + t(H));
        return H;
    };
}

double step_golden_section (phi_function phi, vecd x, vecd d, double error = 1e-6, double rho = 1)
{
    double theta1 = (3 - sqrt(5)) / 2.;
    double theta2 = 1 - theta1;
    // stage 1 - obtaining the interval [a, b]
    double a = 0;
    double s = rho;
    double b = 2 * rho;
    while (phi(x, b, d) < phi(x, s, d))
    {
        a = s;
        s = b;
        b *= 2;
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
    }
    return u / 2. + v / 2.;
}

function<double(double)> D(phi_function phi, vecd x, vecd d, double error = 1e-6)
{
    real_function phi_vec = [phi, x, d](vecd t) {
        return phi(x, t[0], d);
    };
    vector_function deriv = grad(phi_vec, 1, error);
    return [deriv](double t) {
        return deriv({ t })[0];
    };
}

/**/
function<double(double)> D2(phi_function phi, vecd x, vecd d, double error = 1e-6)
{
    vector_function deriv = [phi, x, d, error](vecd t) {
        return vecd({ D(phi, x, d, error)(t[0]) });
    };
    matrix_function deriv2 = hess(deriv, 1, error);
    return [deriv2](double t) {
        return deriv2({ t })[0][0];
    };
}
/**/
/*
function<double(double)> D2(phi_function phi, vecd x, vecd d, double error = 1e-6)
{
    return [phi, x, d, error](double t) {
        return (phi(x, t + 2 * error, d) - 2 * phi(x, t, d) + phi(x, t - 2 * error, d)) / (4 * error * error);
    };
}
/**/

double step_newton_method(phi_function phi, vecd x, vecd d, double t_initial, double error = 1e-6)
{
    double t_previous = -inf;
    double t = t_initial;
    std::function<double(double)> d_phi_dt = D(phi, x, d, error);
    std::function<double(double)> d2_phi_dt2 = D2(phi, x, d);
    while (std::fabs(t - t_previous) > error)
    {
        t_previous = t;
        if (d_phi_dt(t) == 0) break;
        if (d2_phi_dt2(t) == 0) break;
        t = t_previous - d_phi_dt(t_previous) / d2_phi_dt2(t_previous);
    }
    return t;
}

double step_armijo_criterion(real_function func, vector_function grad, vecd x, vecd direction, double eta = 0.5)
{
    double t = 1;
    while (true)
    {
        double linear_delta = eta * t * grad(x) * direction;
        double step = func(x + t * direction);
        if (step <= func(x) + linear_delta)
            break;
        t *= 0.8;
    }
    return t;
}

class Invalid_option_exception: public exception
{
    const char* what()  const noexcept override
    {
        return "Invalid option selected for parameter.";
    }
};

bool check_inequality_restrictions (vector<real_function> restrictions, vecd x)
{
    for (auto restriction : restrictions)
    {   
        //cout << restriction(x) << endl;
        if (restriction(x) > 0) return false;
    }
    return true;
}

void minimize_newton_method (real_function func, int n, vector<real_function> restrictions, 
    vecd x_initial, vecd& x_final, Matrix<double>& trajectory, string linear_search_method = "newton", 
    double sigma = 0, double error = 1e-6, double precision = 1e-6, double rho = 1, double eta = 0.5)
{
    vector_function gradf = grad(func, n, precision);
    matrix_function hessf = hess(gradf, n, precision);

    vecd direction = -gradf(x_initial);
    x_final = x_initial;
    trajectory.push_back(x_final);
    phi_function phi = [func](vecd x, double t, vecd d) {
        return func(x + t * d);
    };

    while (Matrix<double>::norm(gradf(x_final)) > error)
    {
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
            throw Invalid_option_exception();
        if (std::isnan(t_k)) break;

        Matrix<double> M_k = Matrix<double>();
        if (sigma > 0) M_k = (hessf(x_final) + sigma * I<double>(n)).inverse();
        else           M_k = hessf(x_final).inverse();
        direction = -M_k * gradf(x_final);
        while (restrictions.size() > 0 && check_inequality_restrictions(restrictions, x_final + t_k * direction) == false) 
            t_k *= 0.9; // if g_i(x_k) >= 0 for some i, let's try to "walk back" a little on direction d to the feasible region
            if (Matrix<double>::norm(t_k * direction) <= error) // we walked back until we're too close to x_{k-1}
                return;
        x_final += + t_k * direction;

        trajectory.push_back(x_final);
        // safety measure to avoid division by zero
        if (trajectory.n_rows() > 1 && *(trajectory.end() - 1) == *(trajectory.end() - 2))
        {
            cout << "Warning: Premature break." << endl;
            break;
        }
        // if the algorithm is taking too long, occasionally check to see if it's making progress
        if (trajectory.n_rows() % (size_t)1e3 == 0) cout << trajectory.n_rows() << ": " << *(trajectory.end() - 1) << endl;
    }
}

void minimize_conjugate_gradient (real_function func, int n, vector<real_function> restrictions, 
    vecd x_initial, vecd& x_final, Matrix<double>& trajectory, string linear_search_method = "newton", 
    double error = 1e-6, double precision = 1e-6, double rho = 1, double eta = 0.5)
{
    vector_function gradf = grad(func, n, precision);
    matrix_function hessf = hess(gradf, n, precision);

    vecd direction = -gradf(x_initial);
    x_final = x_initial;
    trajectory.push_back(x_final);
    phi_function phi = [func](vecd x, double t, vecd d) {
        return func(x + t * d);
    };

    while (Matrix<double>::norm(gradf(x_final)) > error)
    {
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
            throw Invalid_option_exception();
        if (std::isnan(t_k)) break;
        // if g_i(x_k) >= 0 for some i, let's try to "walk back" a little on direction d to the feasible region
        while (restrictions.size() > 0 && check_inequality_restrictions(restrictions, x_final + t_k * direction) == false) 
            t_k *= 0.9; 
            if (Matrix<double>::norm(t_k * direction) <= error) // we walked back until we're too close to x_{k-1}
                return;
        vecd x_prev = x_final;
        x_final += + t_k * direction;
        double beta_k = (direction * hessf(x_final) * gradf(x_final)) / (direction * hessf(x_final) * direction);
        direction = -gradf(x_final) + beta_k * direction;

        trajectory.push_back(x_final);
        // safety measure to avoid division by zero
        if (trajectory.n_rows() > 1 && *(trajectory.end() - 1) == *(trajectory.end() - 2))
        {
            cout << "Warning: Premature break." << endl;
            break;
        }
        // if the algorithm is taking too long, occasionally check to see if it's making progress
        if (trajectory.n_rows() % (size_t)1e3 == 0) cout << trajectory.n_rows() << ": " << *(trajectory.end() - 1) << endl;
    }
}

void minimize_quasi_newton_method (real_function func, int n, vector<real_function> restrictions, 
    vecd x_initial, vecd& x_final, Matrix<double>& trajectory, string linear_search_method = "newton", 
    double beta = 1, double error = 1e-6, double precision = 1e-6, double rho = 1, double eta = 0.5)
{
    vector_function gradf = grad(func, n, precision);
    matrix_function hessf = hess(gradf, n, precision);

    vecd direction = -gradf(x_initial);
    x_final = x_initial;
    trajectory.push_back(x_final);
    phi_function phi = [func](vecd x, double t, vecd d) {
        return func(x + t * d);
    };
    Matrix<double> H = (beta) * I<double>(n);

    while (Matrix<double>::norm(gradf(x_final)) > error)
    {
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
            throw Invalid_option_exception();
        if (std::isnan(t_k)) break;

        direction = -H * gradf(x_final);
        while (restrictions.size() > 0 && check_inequality_restrictions(restrictions, x_final + t_k * direction) == false) 
            t_k *= 0.9; // if g_i(x_k) >= 0 for some i, let's try to "walk back" a little on direction d to the feasible region
            if (Matrix<double>::norm(t_k * direction) <= error) // we walked back until we're too close to x_{k-1}
                return;
        vecd p = t_k * direction;
        x_final += p;      
        vecd q = gradf(x_final) - gradf(x_final - p);
        H += (1 + (q * H * q) / (p * q)) * ((t(p) * Matrix<double>(p, false)) / (p * q)) - ((t(p) * Matrix<double>(q, false)) * H + H * (t(q) * Matrix<double>(p, false))) / (p * q);

        trajectory.push_back(x_final);
        // safety measure to avoid division by zero
        if (trajectory.n_rows() > 1 && *(trajectory.end() - 1) == *(trajectory.end() - 2))
        {
            cout << "Warning: Premature break." << endl;
            break;
        }
        // if the algorithm is taking too long, occasionally check to see if it's making progress
        if (trajectory.n_rows() % (size_t)1e3 == 0) cout << trajectory.n_rows() << ": " << *(trajectory.end() - 1) << endl;
    }
}

real_function generate_barrier_function (vector<real_function> restrictions, string type = "log")
{
    if (type == "log")
        return [restrictions] (vecd x) {
            double sum = 0;
            for (auto restriction : restrictions)
            {
                sum += log(-restriction(x));
            }
            return -sum;
        };
    else if (type == "inverse")
        return [restrictions] (vecd x) {
            double sum = 0;
            for (auto restriction : restrictions)
            {
                sum += - 1 / restriction(x);
            }
            return -sum;
        };
    else throw Invalid_option_exception();
}

void minimize_barrier_method (real_function func, int n, vector<real_function> restrictions, vecd x_initial, vecd& x_final, double& f_final,
    Matrix<double>& trajectory, string minimization_method = "conjugate-gradient", string linear_search_method = "newton",
    string barrier_type = "log", double mu = 10, double beta = 0.1, double error = 1e-6,
    double precision = 1e-6, double rho = 1, double eta = 0.5, double sigma = 0)
{
    if (!check_inequality_restrictions(restrictions, x_initial))
    {
        cout << "Error: infeasible starting point." << endl << endl;
        return;
    }
    vecd x_prev = x_initial;
    real_function barrier_function = generate_barrier_function(restrictions, barrier_type);
    while (true)
    {
        real_function lagrangian = [func, mu, barrier_function](vecd x) { return func(x) + mu * barrier_function(x); };
        if (minimization_method == "newton-method")
            minimize_newton_method(lagrangian, n, restrictions, x_prev, x_final, trajectory,
                                    linear_search_method, sigma, error, precision, rho, eta);
        else if (minimization_method == "conjugate-gradient")
            minimize_conjugate_gradient(lagrangian, n, restrictions, x_prev, x_final, trajectory,
                                        linear_search_method, error, precision, rho, eta);
        else if (minimization_method == "quasi-newton")
            minimize_quasi_newton_method(lagrangian, n, restrictions, x_initial, x_final, trajectory,
                                         linear_search_method, sigma, error, precision, rho, eta);
        else throw Invalid_option_exception();
        cout << trajectory.n_rows() << ": " << "mu: " << mu << ", B(x): " << barrier_function(x_final) << ", mu*B(x): " << mu * barrier_function(x_final);
        cout << ", g(x) <= 0: " << ((check_inequality_restrictions(restrictions, x_final)) ? "true" : "false") << endl;
        if (std::abs(mu * barrier_function(x_final)) < error) 
            break;
        mu *= beta;
        x_prev = x_final;
    }
    f_final = func(x_final);   
}

/* 
General form of a penalty function:
\alpha(x) = \sum_{i = 1}^m \phi(g_i(x)) + \sum_{i = 1}^l \psi (h_i(x))
where
\phi(y) = 0 if y <= 0 and \phi(y) > 0 if y > 0
\psi(y) = 0 if y = 0  and \psi(y) > 0 if y != 0
*/
real_function generate_penalty_function (vector<real_function> inequality_restrictions, vector<real_function> equality_restrictions, unsigned int power = 2)
{
    function<double(double)> phi = [power](double y) { return std::pow(std::max(0., y), power); };
    function<double(double)> psi;
    if (power % 2)
        psi = [power](double y) { return std::pow(std::abs(y), power); };
    else
        psi = [power](double y) { return std::pow(y, power); };
    return [phi, psi, equality_restrictions, inequality_restrictions](vecd x) {
        double sum = 0;
        for (auto restriction : inequality_restrictions)
            sum += phi(restriction(x));
        for (auto restriction : equality_restrictions)
            sum += psi(restriction(x));
        return sum;
    };
}

void minimize_penalty_method (real_function func, int n, vector<real_function> inequality_restrictions, vector<real_function> equality_restrictions, vecd x_initial,
    vecd& x_final, double& f_final, Matrix<double>& trajectory, string minimization_method = "conjugate-gradient", string linear_search_method = "newton",
    double power = 2, double mu = 0.1, double beta = 10, double error = 1e-6, double precision = 1e-6, double rho = 1, double eta = 0.5, double sigma = 0)
{
    auto dont_check_restrictions = vector<real_function>();
    vecd x_prev = x_initial;
    real_function penalty_function = generate_penalty_function(inequality_restrictions, equality_restrictions, power);
    while (true)
    {
        real_function lagrangian = [func, mu, penalty_function](vecd x) { return func(x) + mu * penalty_function(x); };
        if (minimization_method == "newton-method")
            minimize_newton_method(lagrangian, n, dont_check_restrictions, x_prev, x_final, trajectory,
                                    linear_search_method, sigma, error, precision, rho, eta);
        else if (minimization_method == "conjugate-gradient")
            minimize_conjugate_gradient(lagrangian, n, dont_check_restrictions, x_prev, x_final, trajectory,
                                        linear_search_method, error, precision, rho, eta);
        else if (minimization_method == "quasi-newton")
            minimize_quasi_newton_method(lagrangian, n, dont_check_restrictions, x_initial, x_final, trajectory,
                                         linear_search_method, sigma, error, precision, rho, eta);
        else throw Invalid_option_exception();
        cout << trajectory.n_rows() << ": " << "mu: " << mu << ", a(x): " << penalty_function(x_final) << ", mu*a(x): " << mu * penalty_function(x_final) << endl;
        if (std::abs(mu * penalty_function(x_final)) < error) 
            break;
        mu *= beta;
        x_prev = x_final;
    }
    f_final = func(x_final);  
}

/*
void minimize_newton_method (real_function func, int n, vecd x_initial, vecd& x_final, Matrix<double>& trajectory, 
    string linear_search_method = "newton", double sigma = 0, double error = 1e-6, double precision = 1e-6, 
    double rho = 1, double eta = 0.5)
{
    minimize_newton_method (func, n, no_restrictions, x_initial, x_final, trajectory, linear_search_method, 
    sigma, error, precision, rho, eta);
}

void minimize_conjugate_gradient (real_function func, int n, vecd x_initial, vecd& x_final, Matrix<double>& trajectory, 
    string linear_search_method = "newton", double error = 1e-6, double precision = 1e-6, double rho = 1, double eta = 0.5)
{
    minimize_conjugate_gradient (func, n, no_restrictions, x_initial, x_final, trajectory, linear_search_method, 
    error, precision, rho, eta);
}

void minimize_quasi_newton_method (real_function func, int n, vecd x_initial, vecd& x_final, Matrix<double>& trajectory, 
    string linear_search_method = "newton", double beta = 1, double error = 1e-6, double precision = 1e-6, double rho = 1, 
    double eta = 0.5)
{
    minimize_quasi_newton_method (func, n, no_restrictions, x_initial, x_final, trajectory, linear_search_method, 
    beta, error, precision, rho, eta);
}
/**/
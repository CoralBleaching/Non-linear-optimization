#include <iostream>
#include "restricted-optimization.hh"

using namespace::std;

typedef struct {
    vecd x_final;
    Matrix<double> trajectory;
} test_data;

test_data execute_test (real_function function, int n, vector<real_function> inequality_restrictions, vector<real_function> equality_restrictions, 
    vecd x_initial, double mu, double beta, string mode = "penalty-method", string minimization_method = "cojugate-gradient", string linear_search_method = "golden-section", 
    string barrier_type = "log", unsigned int power = 2, double error = 1e-6, double precision = 1e-6, 
    double rho = 1, double eta = 0.5, double sigma = 0)
{
    vecd x_final;
    double f_final;
    Matrix<double> trajectory;

    vector_function gradf = grad(function, n);

    if (mode == "barrier-method")
        minimize_barrier_method(function, n, inequality_restrictions, x_initial, x_final, f_final, trajectory, minimization_method, linear_search_method,
                            barrier_type, mu, beta, error, precision, rho, eta, sigma);
    else if (mode == "penalty-method")
        minimize_penalty_method(function, n, inequality_restrictions, equality_restrictions, x_initial, x_final, f_final, trajectory,
        minimization_method, linear_search_method, power, mu, beta, error, precision, rho, eta, sigma);
    else throw Invalid_option_exception();

    int n_steps = trajectory.n_rows();
    if (n_steps == 0) return {x_final, trajectory};
    int i = (n_steps > 300) ? n_steps - 300 : 0;
    for (; i < n_steps; i++)   
        std::cout << i << ": " << trajectory[i] << endl;
    std::cout << "x_final = " << x_final << endl << "f_final = " << f_final << endl;
    std::cout << "steps = " << n_steps << endl;
    cout << "||grad(x)||: " << Matrix<double>::norm(gradf(x_final)) << endl;
    cout << "g(x) < 0: " << ((check_inequality_restrictions(inequality_restrictions, x_final)) ? "true" : "false") << endl << endl;
    return { x_final, trajectory };
}

// test 1 variables declaration (hard way)
vecd x_initial1 = { -2, 1 };
double function1 (vecd x)
{
    return 100 * pow(x[1] - pow(x[0], 2), 2) + pow(1 - x[0], 2);
}
double restriction11 (vecd x)
{
    return x[0] - x[1] * x[1];
}
double restriction12 (vecd x)
{
    return -(x[0] * x[0]) + x[1];
}
vector<real_function> restrictions1 = { restriction11, restriction12 };

/* NOTE: restrictions are always of the form "g_i(x) <= 0"!
*/
int main()
{   
    // BARRIER
    // test 1
    /**
    execute_test(function1, 2, no_restrictions, no_restrictions, x_initial1, 10, 0.1, "barrier-method", "conjugate-gradient", "golden-section", "log");
    execute_test(function1, 2, restrictions1, no_restrictions, x_initial1, 10, 0.1, "barrier-method", "newton-method", "golden-section", "log");
    /**/
    // test 2 
    auto function2 = [](vecd x) { 
        return x[0] * x[0] + 0.5 * x[1] * x[1] + x[2] * x[2] + 0.5 * x[3] * x[3] - x[0] * x[2] + x[2] * x[3] - x[0] - 3 * x[1] + x[2] - x[3]; 
    };
    vector<real_function> restrictions2 = 
    {
        [](vecd x) { return -5 + x[0] + 2*x[1] + x[2] + x[3]; },
        [](vecd x) { return -4 + 3*x[0] + x[1] + 2*x[2] - x[3]; },
        [](vecd x) { return -x[1] - 4*x[3] + 1.5; },
        [](vecd x) { return -x[0];},
        [](vecd x) { return -x[1];},
        [](vecd x) { return -x[2];},
        [](vecd x) { return -x[3];}
    };
    vecd x_initial2 = { 0.5, 0.5, 0.5, 0.5 };
    /**
    cout << "test 2" << endl;
    execute_test(function2, 4, no_restrictions, no_restrictions, x_initial2, 10, 0.1, "barrier-method", "conjugate-gradient", "armijo");
    execute_test(function2, 4, restrictions2, no_restrictions, x_initial2, 10, 0.1, "barrier-method", "conjugate-gradient", "newton");
    /**/
    // test 3
    real_function function3 = [](vecd x) { return pow(x[0] - 10, 3) + pow(x[1] - 20, 3); };
    vector<real_function> restrictions3 =
    {
        [](vecd x) { return -pow(x[0] - 5, 2) - pow(x[1] - 5, 2) + 100; },
        [](vecd x) { return -pow(x[0] - 6, 2) - pow(x[1] - 5, 2); }, // 0-radius circle -> meaningless restriction
        [](vecd x) { return -82.81 + pow(x[0] - 6, 2) + pow(x[1] - 5, 2); }
    };
    vecd x_initial3 = { 13, 0 };
    /* NOTE:
    feasible region: 14.095 < x < 15, 1/10 (50 - sqrt(-100 x^2 + 1200 x + 4681)) <= y <= 5 - sqrt(-x^2 + 10 x + 75)
    example feasible starting point: { 14.5, 1.8139428194732083 }
    */
    /**
    cout << "test 3" << endl;
    execute_test(function3, 2, no_restrictions, no_restrictions, x_initial3, 10, 0.1, "barrier-method", "newton-method", "armijo", "log", 2, 1e-10, 1e-6, 1, 0.9, 0);
    execute_test(function3, 2, restrictions3, no_restrictions, x_initial3, 10, 0.1, "barrier-method", "newton-method", "armijo", "log", 2, 1e-6, 1e-6, 1, 0.5, 0);
    execute_test(function3, 2, restrictions3, no_restrictions, { 14.115, 0.885 }, 10, 0.1, "barrier-method", "newton-method", "armijo", "log", 2, 1e-10, 1e-6, 1, 0.9, 0);

    /**/

    // PENALTY
    // test 1
    auto function4 = [](vecd x) { return 0.01 * pow(x[0] - 1, 2) + pow(x[1] - pow(x[0], 2), 2); };
    vector<real_function> equality_restrictions4 = {
        [](vecd x) { return x[0] + pow(x[2], 2) + 1; }
    };
    vecd x_initial4 = { 2, 2, 2 };
    /**
    cout << "penalty test 1" << endl;
    execute_test(function4, 3, no_restrictions, no_restrictions, x_initial4, 0.1, 10, "penalty-method", "conjugate-gradient", "armijo");
    execute_test(function4, 3, no_restrictions, equality_restrictions4, x_initial4, 0.1, 10,  "penalty-method", "conjugate-gradient", "armijo");
    /**/

    //test 2
    auto function5 = [](vecd x) { return pow(x[0] - x[1], 2) + pow(x[1] + x[2] - 2, 2) + pow(x[3] - 1, 2) + pow(x[4] - 1, 2); };
    vector<real_function> equality_restrictions5 = {
        [](vecd x) { return x[0] + 3*x[1] - 4; },
        [](vecd x) { return x[2] + x[3] - 2*x[4]; },
        [](vecd x) { return x[1] - x[4]; }
    };
    vecd x_initial5 = { 2.5, 0.5, 2, -1, 0.5 };

    /**
    cout << "penalty test 2" << endl;
    execute_test(function5, 5, no_restrictions, no_restrictions, x_initial5, 0.1, 10, "penalty-method",
                 "conjugate-gradient", "newton", "log", 2, 1e-6, 1e-6, 1, 0.5, 1);    
    test_data data5 = execute_test(function5, 5, no_restrictions, equality_restrictions5, x_initial5, 0.1, 10, "penalty-method",
                 "conjugate-gradient", "quadratic", "log", 2, 1e-10, 1e-6, 1, 0.5, 1);
    // checking for restrictions
    cout << "h(x): { ";
    for (auto restriction : equality_restrictions5)
        cout << restriction(data5.x_final) << " ";
    cout << "}" << endl << endl;
    /**/

    //test 3
    auto function6 = [](vecd x) { return pow(x[0] - 2, 2) + pow(x[1] - 1, 2); };
    vector<real_function> inequality_restrictions6 = { [](vecd x) { return 0.25*x[0]*x[0] + x[1]*x[1] - 1; } };
    vector<real_function> equality_restrictions6 = { [](vecd x) { return x[0] - 2*x[1] + 1; } };
    vecd x_initial6 = { 2, 2 };

    /**
    cout << "penalty test 3" << endl;
    execute_test(function6, 2, no_restrictions, no_restrictions, x_initial6, 0.1, 10, "penalty-method",
                 "conjugate-gradient", "newton", "log", 2, 1e-6, 1e-6, 1, 0.5, 1);    
    test_data data6 = execute_test(function6, 2, inequality_restrictions6, equality_restrictions6, x_initial6, 0.1, 10, "penalty-method",
                 "conjugate-gradient", "quadratic", "log", 2, 1e-10, 1e-6, 1, 0.5, 1);
    // checking for restrictions
    cout << "h(x): { ";
    for (auto restriction : equality_restrictions6)
        cout << restriction(data6.x_final) << " ";
    cout << "}" << endl << endl;
    /**/

    // PENALTY AND BARRIER
    
    //test 1
    auto function7 = [](vecd x) { return x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]; };
    vector<real_function> inequality_restrictions7 = {
        [](vecd x) { return -(x[0]*x[1]*x[2]*x[3]) + 25; },
        [](vecd x) { return -(x[0]*x[0]) - (x[1]*x[1]) - (x[2]*x[2]) - (x[3]*x[3]) + 40; },
        [](vecd x) { return -x[0] + 1; },
        [](vecd x) { return -x[1] + 1; },
        [](vecd x) { return -x[2] + 1; },
        [](vecd x) { return -x[3] + 1; },
        [](vecd x) { return x[0] - 5; },
        [](vecd x) { return x[1] - 5; },
        [](vecd x) { return x[2] - 5; },
        [](vecd x) { return x[3] - 5; },
    };
    vecd x_initial7 = { 3, 4, 4, 3};

    /**
    cout << "both test 1" << endl;
    execute_test(function7, 4, no_restrictions, no_restrictions, x_initial7, 0.1, 10, "barrier-method",
                 "conjugate-gradient", "armijo", "log", 2, 1e-6, 1e-6, 1, 0.9, 0); 
    /**/
    /**
    execute_test(function7, 4, inequality_restrictions7, no_restrictions, x_initial7, 10, 0.1, "barrier-method",
                 "conjugate-gradient", "armijo", "log", 2, 1e-6, 1e-6, 1, 0.5, 0);
    /**/  
    /**
    test_data data7 = execute_test(function7, 4, inequality_restrictions7, no_restrictions, x_initial7, 1, 1.000001, "penalty-method",
                 "conjugate-gradient", "armijo", "log", 2, 1e-6, 1e-6, 1, 0.5, 10);  
    // checking for restrictions
    cout << "h(x): { ";
    for (auto restriction : equality_restrictions6)
        cout << restriction(data7.x_final) << " ";
    cout << "}" << endl << endl;  
    /**/
    
    auto function8 = [](vecd x) { return x[0] - x[1] - x[2] - x[0]*x[2] + x[0]*x[3] + x[1]*x[2] - x[1]*x[3]; };
    vector<real_function> inequality_restrictions8 = {
        [](vecd x) { return -8 + x[0] + 2*x[1]; },
        [](vecd x) { return -12 + 4*x[0] + x[1]; },
        [](vecd x) { return -12 + 3*x[0] + 4*x[1]; },
        [](vecd x) { return -8 + 2*x[2] + x[3]; },
        [](vecd x) { return -8 + x[2] + 2*x[3]; },
        [](vecd x) { return -5 + x[2] + x[3]; },
        [](vecd x) { return -x[0]; },
        [](vecd x) { return -x[1]; },
        [](vecd x) { return -x[2]; },
        [](vecd x) { return -x[3]; },
    };
    vecd x_initial8 = { 1, 1, 1, 1 };
    /**
    cout << "both test 2" << endl;
    test_data data81 = execute_test(function8, 4, no_restrictions, no_restrictions, x_initial8, 10, 0.1, "barrier-method",
                 "quasi-newton", "newton", "log", 2, 1e-6, 1e-6, 1, 0.5, 1);
    cout << "g(x) <= 0: " << check_inequality_restrictions(inequality_restrictions8, data81.x_final) << endl; 
    cout << "g(x): { ";
    for (auto restriction : inequality_restrictions8)
        cout << restriction(data81.x_final) << " ";
    cout << "}" << endl << endl;  
    /**/
    /**
    execute_test(function8, 4, inequality_restrictions8, no_restrictions, x_initial8, 1, 0.1, "barrier-method",
                 "quasi-newton", "newton", "log", 2, 1e-6, 1e-6, 1, 0.5, 0);
    execute_test(function8, 4, inequality_restrictions8, no_restrictions, x_initial8, 1, 0.1, "barrier-method",
                 "conjugate-gradient", "armijo", "inverse", 2, 1e-6, 1e-6, 1, 0.5, 0);
    test_data data82 = execute_test(function8, 4, inequality_restrictions8, no_restrictions, x_initial8, 1, 0.1, "barrier-method",
                 "conjugate-gradient", "armijo", "log", 2, 1e-6, 1e-6, 1, 0.5, 0);
    cout << "g(x) <= 0: " << check_inequality_restrictions(inequality_restrictions8, data81.x_final) << endl; 
    cout << "g(x): { ";
    for (auto restriction : inequality_restrictions8)
        cout << restriction(data82.x_final) << " ";
    cout << "}" << endl << endl;  
    /**/  
    /**
    test_data data83 = execute_test(function8, 4, inequality_restrictions8, no_restrictions, x_initial8, 0.1, 10, "penalty-method",
                 "conjugate-gradient", "armijo", "log", 20, 1e-5, 1e-6, 1, 0.1, 10);  
    // checking for restrictions
    cout << "h(x): { ";
    for (auto restriction : inequality_restrictions8)
        cout << restriction(data83.x_final) << " ";
    cout << "}" << endl << endl;  
    /**/

    auto function9 = [](vecd x) { 
        return pow(x[0] - 10, 2) + 5*pow(x[1] - 12, 2) + pow(x[3], 4) + 3*pow(x[3] - 11, 2) 
               + 10*pow(x[4], 6) + 7*pow(x[5], 2) + pow(x[6], 4) - 4*x[5]*x[6] - 10*x[5] - 8*x[6]; 
    };
    vector<real_function> inequality_restrictions9 = {
        [](vecd x) { return - (127 -x[0]*x[0] - 3*pow(x[1], 4) - x[2] - 4*x[3]*x[3] - 5*x[4]); },
        [](vecd x) { return - (282 - 7*x[0] - 3*x[1] - 10*x[2]*x[2] - x[3] + x[4]); },
        [](vecd x) { return - (196 - 23*x[0]*x[0] - x[1]*x[1] - 6*x[5]*x[5] + 8*x[6]); },
        [](vecd x) { return - (-4*x[0]*x[0] - x[1]*x[1] + 3*x[0]*x[1] - 2*x[2]*x[2] - 5*x[5] + 11*x[6]); }
    };
    vecd x_initial9 = { 1, 2, 0, 4, 0, 1, 1};
    /**
    cout << "both test 3" << endl;
    test_data data91 = execute_test(function9, 7, no_restrictions, no_restrictions, x_initial9, 10, 0.1, "barrier-method",
                 "conjugate-gradient", "armijo", "log", 2, 1e-6, 1e-6, 1, 0.5, 1);
    cout << "g(x) <= 0: " << check_inequality_restrictions(inequality_restrictions9, data91.x_final) << endl; 
    cout << "g(x): { ";
    for (auto restriction : inequality_restrictions8)
        cout << restriction(data91.x_final) << " ";
    cout << "}" << endl << endl;  
    /**/
    /**
    execute_test(function9, 7, inequality_restrictions9, no_restrictions, x_initial9, 10, 0.1, "barrier-method",
                 "conjugate-gradient", "quadratic", "log", 2, 1e-6, 1e-6, 1, 0.5, 0);
    execute_test(function9, 7, inequality_restrictions9, no_restrictions, x_initial9, 1, 0.1, "barrier-method",
                 "conjugate-gradient", "armijo", "inverse", 2, 1e-6, 1e-6, 1, 0.5, 0);
    test_data data92 = execute_test(function9, 7, inequality_restrictions9, no_restrictions, x_initial9, 1, 0.1, "barrier-method",
                 "conjugate-gradient", "armijo", "log", 2, 1e-6, 1e-6, 1, 0.5, 0);
    cout << "g(x) <= 0: " << check_inequality_restrictions(inequality_restrictions8, data92.x_final) << endl; 
    cout << "g(x): { ";
    for (auto restriction : inequality_restrictions9)
        cout << restriction(data92.x_final) << " ";
    cout << "}" << endl << endl;  
    /**/  
    /**/
    test_data data93 = execute_test(function9, 7, inequality_restrictions9, no_restrictions, x_initial9, 0.1, 10, "penalty-method",
                 "conjugate-gradient", "armijo", "log", 2, 1e-5, 1e-6, 1, 0.1, 0);  
    // checking for restrictions
    cout << "h(x): { ";
    for (auto restriction : inequality_restrictions9)
        cout << restriction(data93.x_final) << " ";
    cout << "}" << endl << endl;  
    /**/
}

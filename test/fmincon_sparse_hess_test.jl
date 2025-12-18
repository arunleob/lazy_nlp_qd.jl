let

    function cost(params, x)
        return 0.5*x'*params.Q*x + params.q'*x
    end

    function cost_gradient!(params, grad, x)
        grad .= params.Q*x + params.q 
        return nothing
    end

    function constraint(x)
        return [(x - [2; 0])'*(x - [2; 0]);]
    end

    function constraint!(params, cval, x)
        cval .= constraint(x)
        return nothing 
    end

    function constraint_jacobian!(params, conjac, x)
        conjac .= FD.jacobian(constraint, x)
        return nothing 
    end

    function lagrangian_hessian!(params, hess, x, σ, λ)
        hess .= sparse(σ*params.Q + FD.jacobian(_x -> FD.jacobian(constraint, _x)'*λ, x))
        return nothing 
    end
        
    nx, nc = 2, 1

    # primal variable bounds 
    x_l = -Inf * ones(nx)
    x_u =  Inf * ones(nx)

    # constriant bounds
    c_l = [-Inf;]
    c_u = [1;]

    # sparse jacobian matrix with correct sparsity pattern
    temp_con_jac = sparse(FD.jacobian(constraint, ones(nx)))
    temp_lag_hess = sparse(I(2) + FD.jacobian(_x -> FD.jacobian(constraint, _x)'*ones(1), ones(2)))
        
    # things I need for my functions 
    params = (
        Q = I(2), 
        q = zeros(2), 
    )

    # initial guess
    x0 = .1*randn(nx)

    x = lazy_nlp_qd.sparse_fmincon(cost::Function,
                                    cost_gradient!::Function,
                                    constraint!::Function,
                                    constraint_jacobian!::Function,
                                    temp_con_jac,
                                    lagrangian_hessian!::Function,
                                    temp_lag_hess,
                                    x_l::Vector,
                                    x_u::Vector,
                                    c_l::Vector,
                                    c_u::Vector,
                                    x0::Vector,
                                    params::NamedTuple;
                                    tol = 1e-4,
                                    c_tol = 1e-4,
                                    max_iters = 1_000,
                                    print_level = 5)

    @test norm(x - [1; 0]) < 1e-3
end

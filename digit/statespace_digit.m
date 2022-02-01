function [qdot] = statespace_digit(q0, u, model)
    % This computes the xdot = [dq;ddq] given a controller input `u` and
    % states `x=[q;dq]`. 
    
    % extract the state variables into x and dx
    nq = model.Dimension;
    q = q0(1:nq);
    dq = q0(nq+1:end);
        
    % compute the mass matrix and drift vector (internal dynamics)
    M = calcMassMatrix(model, q);
    Fv = calcDriftVector(model, q, dq);
    
    %% NOTE: 
    % here we assume no external input, and the joint torques are given by
    % the input arguments `u`. Therefore no need to call the callback
    % function. 
    torque = model.Inputs.torque;
    Gmap = feval(torque.Gmap.Name,q);
    Gv   = Gmap*u;
    
    ddq_free = M\(-Fv+Gv);
    
    %% holonomic constraints
    h_cstr_name = fieldnames(model.HolonomicConstraints);
    if ~isempty(h_cstr_name)           % if holonomic constraints are defined
        h_cstr = struct2array(model.HolonomicConstraints);
        n_cstr = length(h_cstr);
        % determine the total dimension of the holonomic constraints
        cdim = sum([h_cstr.Dimension]);
        % initialize the Jacobian matrix
        Je = zeros(cdim,nq);
        Jedot = zeros(cdim,nq);
        
        idx = 1;
        for i=1:n_cstr
            cstr = h_cstr(i);
            
            % calculate the Jacobian
            [Jh,dJh] = calcJacobian(cstr,q,dq);
            cstr_indices = idx:idx+cstr.Dimension-1;
                    
            Je(cstr_indices,:) = Jh;
            Jedot(cstr_indices,:) = dJh; 
            idx = idx + cstr.Dimension;
        end 
    else
        Je = [];
        Jedot = [];
    end
    
    if isempty(Je)
        ddq = ddq_free;
    else
        Xi = Je * (M \ transpose(Je));
        P = eye(nq) - M\(transpose(Je)/Xi)*Je;
        % ddq = M\(-Fv+Gv+Je'*lambda);
        ddq = P*ddq_free;
    end
    
    % the system dynamics
    qdot = [dq; ddq];
end


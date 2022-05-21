function [qnew] = pend_push(q0, T, push)
    
    qnew = q0;

    if (isempty(push))
        return;
    end

    for i = 1:length(push(:,1))
        if (push(i,1) == T)
            qnew(2*push(i,2)) = qnew(2*push(i,2)) + push(i,3);
        end
    end

end
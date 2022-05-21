function [anim] = animation_3link(q, T, m, L)
    %% body parameters.
    m1 = m(1); m2 = m(2); m3 = m(3);
    L1 = L(1); L2 = L(2); L3 = L(3);
    
    %% unpacking
    % adj = pi/2;
    theta1List = q(:,1);
    theta2List = q(:,3);
    theta3List = q(:,5);
    
    CoM = map_CoM(q,m,L);
    xCoM = CoM(:,1);
    yCoM = CoM(:,2);
    
    %% Delay
    dt = T(2) - T(1);

    %% animation
    anim = figure(100);clf;
    anim.Position = [0 0 400 800];
    hold on
    for iFrame = 1:length(q(:,1))
        th1 = theta1List(iFrame);
        th2 = theta2List(iFrame);
        th3 = theta3List(iFrame);

        xAnkle = 0; yAnkle = 0;
        xKnee = xAnkle + L1*cos(th1); yKnee = yAnkle + L1*sin(th1);
        xHip = xKnee + L2*cos(th1+th2); yHip = yKnee + L2*sin(th1+th2);
        xHead = xHip + L3*cos(th1+th2+th3); yHead = yHip + L3*sin(th1+th2+th3);
        if iFrame == 1
            pShin = plot([xAnkle xKnee],[yAnkle yKnee],'r','linewidth',3); hold on;
            pThigh = plot([xKnee xHip],[yKnee yHip],'b','linewidth',3); hold on;
            pTorso = plot([xHip xHead],[yHip yHead],'color',[0.4660 0.6740 0.1880],'linewidth',3); hold on;
            pKnee = plot(xKnee, yKnee, '.', 'color', 'k', 'markersize', m1); hold on;
            pHip = plot(xHip, yHip, '.', 'color', 'k', 'markersize', m2); hold on;
            pHead = plot(xHead, yHead, '.', 'color', 'k', 'markersize', m3); hold on;
            pCoM = plot(xCoM(iFrame), yCoM(iFrame), '-s', 'markersize', 8, 'color', '#7E2F8E', 'markerfacecolor', '#7E2F8E'); hold on;

            plot([-L1/2 L1/2],[0 0],'color',[0 0 0],'linewidth',2);
        else
            pShin.XData = [xAnkle xKnee];
            pShin.YData = [yAnkle yKnee];
            pThigh.XData = [xKnee xHip];
            pThigh.YData = [yKnee yHip];
            pTorso.XData = [xHip xHead];
            pTorso.YData = [yHip yHead];
            pKnee.XData  = xKnee;
            pKnee.YData = yKnee;
            pHip.XData = xHip;
            pHip.YData = yHip;
            pHead.XData = xHead;
            pHead.YData = yHead;
            pCoM.XData = xCoM(iFrame);
            pCoM.YData = yCoM(iFrame);
            
        end
        axis equal; % do axis equal before mentioning the xlim ylim
        xlim([-(L1+L2) (L1+L2)]); ylim([-0.5 (L1+L2+L3+0.5)]);
        pause(dt/10);
        drawnow;
%         hold off
    end
end
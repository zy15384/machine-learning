function[] = Exam1_Q2(xt,yt,sigma)
%picture1    
subplot(2,2,1);
Q2_draw_gradient(1,xt,yt,[1],[0],sigma);
hold on
plot(1,0,'ro'); % plot the landmark locations
hold on
plot(xt,yt,'k+'); % plot the true locations of the objective
axis([-2 2 -2 2]);
xlabel("x");
ylabel("y");
legend("gradient",'landmark',"real point");
title("one landmark");
grid minor;
%picture2
subplot(2,2,2);
Q2_draw_gradient(2,xt,yt,[1,-1],[0,0],sigma);
hold on
plot([1,-1],[0,0],'ro'); % plot the landmark locations
hold on
plot(xt,yt,'k+'); % plot the true locations of the objective
axis([-2 2 -2 2]);
xlabel("x");
ylabel("y");
legend("gradient",'landmark',"real point");
title("two landmark");
grid minor;
%picture3
subplot(2,2,3);
Q2_draw_gradient(3,xt,yt,[1, -0.5, -0.5],[0, -1.732/2, 1.732/2],sigma);
hold on
plot([1,-0.5,-0.5],[0,-1.732/2,1.732/2],'ro'); % plot the landmark locations
hold on
plot(xt,yt,'k+'); % plot the true locations of the objective
axis([-2 2 -2 2]);
xlabel("x");
ylabel("y");
legend("gradient",'landmark',"real point");
title("three landmark");
grid minor;
%picture4
subplot(2,2,4);
Q2_draw_gradient(4,xt,yt,[1,-1,0,0],[0,0,1,-1],0.3)
hold on
plot([1,0,0,-1],[0,-1,1,0],'ro'); % plot the landmark locations
hold on
plot(xt,yt,'k+'); % plot the true locations of the objective
axis([-2 2 -2 2]);
xlabel("x");
ylabel("y");
legend("gradient",'landmark',"real point");
title("four landmark");
grid minor;
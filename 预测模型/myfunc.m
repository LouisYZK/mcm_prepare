% 一般费多项式拟合
function F = myfunc(x,xdata)
    % x输入参数预估值
    F = x(1)*xdata.^2 +x(2)*sin(xdata) + x(3)*xdata.^3;
end
% 调用：
% xdata=［3.6 7.7 9.3 4.1 8.6 2.8 1.3 7.9 10.0 5.4];
% ydata=［16.5 150.6 263.1 24.7 208.5 9.9 2.7 163.9 325.0 54.3；
% x0=［10，10，10］；
% ［x，resnorm］=lsqcurvefit（@myfun，x0，xdata，ydata）
hist(mean(log(abs(randn(10000,10000)))), 100);

mean(mean(log(abs(randn(1000,1000)))))


x = 1:1000
ss = 100./x;
y = exp(1./x+4);
plot(x,y)

orig_ss = 1;
orig_mu = -1;
x = 1:1000;
ss = orig_ss*x;
mu = orig_mu * x;
%y = (exp(ss) - 1).*exp(2*mu +ss);
y = exp(mu + ss./2);
plot(x,y)


x = 1:1000;
ss = 100./x;
mu = -0.6 * x;
y = exp(mu) - 1).*exp(2*mu +ss);
plot(x,y)

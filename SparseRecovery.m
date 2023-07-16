                                        %% HW5
file = load("hw5.mat");
D = file.D;
x = file.x;
[M,N] = size(D);

        %% Part 1 - alef)
tic
N0 = 3;
[Er_1,s_1] = subset_selection(x,D,N0);
elapsed_time_subset_selection = toc;
figure(1);
stem(1:N,s_1);
xlabel("Source Number");
ylabel("Source Amplitude");
title("Sparse Signal Recovery Result / Subset Selection");
grid on;

        %% Part 2 - be)
D_pseudo_inv = transpose(D) * inv(D * transpose(D));
s_2 = D_pseudo_inv * x;
Er_2 = norm(x - D * s_2);
figure(2);
stem(1:N,s_2);
xlabel("Source Number");
ylabel("Source Amplitude");
title("Sparse Signal Recovery Result / Norm-2 Minimization");
grid on;

        %% Part 3 - jim)
thr = 1e-1;
    %% With use of N0
N0 = 3;
tic
[Er_3,s_3] = matching_pursuit(x,D,N0,thr);
elapsed_time_mpn0 = toc;
figure(3);
stem(1:N,s_3);
xlabel("Source Number");
ylabel("Source Amplitude");
title("Sparse Signal Recovery Result / Matching Pursuit with N_0");
grid on;

    %% With not use of  N0
tic
[Er_4,s_4] = matching_pursuit(x,D,0,thr);
elapsed_time_mpthr = toc;
figure(4);
stem(1:N,s_4);
xlabel("Source Number");
ylabel("Source Amplitude");
title("Sparse Signal Recovery Result / Matching Pursuit without N0");
grid on;

        %% Part 4 - dal)
    %% With use of N0
tic
N0 = 3;
[Er_5,s_5] = OMP(x,D,N0,thr);
elapsed_time_ompn0 = toc;
figure(5);
stem(1:N,s_5);
xlabel("Source Number");
ylabel("Source Amplitude");
title("Sparse Signal Recovery Result / Orthogonal Matching Pursuit with N0");
grid on;
    
    %% With not use of N0
tic
thr = 0.1;
[Er_6,s_6] = OMP(x,D,0,thr);
elapsed_time_ompth = toc;
figure(6);
stem(1:N,s_6);
xlabel("Source Number");
ylabel("Source Amplitude");
title("Sparse Signal Recovery Result / Orthogonal Matching Pursuit without N0");
grid on;
        
        %% Part 5 - he)
tic
[Er_7,s_7] = Basis_Pursuit(x,D);
elapsed_time_bp = toc;
figure(7);
stem(1:N,s_7);
xlabel("Source Number");
ylabel("Source Amplitude");
title("Sparse Signal Recovery Result / Basis Pursuit");
grid on;

        %% Part 6 - vav)
tic
[Er_8,s_8] = IRLS(x,D);
elapsed_time_irls = toc;
figure(8);
stem(1:N,s_8);
xlabel("Source Number");
ylabel("Source Amplitude");
title("Sparse Signal Recovery Result / IRLS");
grid on;


        %% Local Necessary Functions
function [er,s] = subset_selection(x,D,N0)
    [~,N] = size(D);
    % C = combntns(1:N,N0);
    C = nchoosek(1:N,N0);
    error = zeros(length(C),1);
    S = zeros(length(C),N,1);
    for i = 1:length(C)
       s = zeros(N,1);
       idx = C(i,:);
       Ds = D(:,idx);
       Ds_psuedo_inv = inv(transpose(Ds)*Ds)*transpose(Ds);
       s_nz = Ds_psuedo_inv * x;
       s(idx) = s_nz;
       S(i,:,:) = s;
       Er = norm(x - D * s);
       error(i) = Er;
    end
    [er,index] = min(error);
    s = S(index,:,:);
    s = transpose(s);
end

function [er,s] = matching_pursuit(x,D,N0,thr)
    [~,N] = size(D);
    if N0 ~= 0
       xr = x;
        s = zeros(N,1);
        for i=1:N0
           corr = transpose(xr) * D;
           [value,idx] = max(corr);
           di = D(:,idx);
           s(idx) = value;
           xr = xr - (transpose(xr)*di)*di;
        end
        er = norm(x - D*s);
    else
        xr = x;
        s = zeros(N,1);
        stopping_criteria = norm(x-D*s)/norm(x);
        i = 0;
        while stopping_criteria > thr
           corr = transpose(xr) * D;
           [value,idx] = max(corr);
           di = D(:,idx);
           s(idx) = value;
           xr = xr - (transpose(xr)*di)*di;
           stopping_criteria = norm(x-D*s)/norm(x);
           i = i+1;
           if i == N
               break
           end
        end
        er = norm(x - D*s);
    end
end

function [er,s] = OMP(x,D,N0,thr)
    [~,N] = size(D);
    if N0 ~= 0
        xr = x;
        s = zeros(N,1);
        D_temp = [];
        nz_idx = [];
        for i=1:N0
           corr = transpose(xr) * D;
           [~,idx] = max(corr);
           di = D(:,idx);
           nz_idx(i) = idx;
           D_temp(:,i) = di;
           if i>=2
              Dtemp_psuedo_inv = inv(transpose(D_temp)*D_temp)*transpose(D_temp);
              s_nz = Dtemp_psuedo_inv * x;
           end
           xr = xr - (transpose(xr)*di)*di;
        end
        s(nz_idx) = s_nz;
        er = norm(x - D*s);
    else
        xr = x;
        s = zeros(N,1);
        D_temp = [];
        nz_idx = [];
        stopping_criteria = norm(x-D*s)/norm(x);
        i = 1;
        while stopping_criteria > thr
           corr = transpose(xr) * D;
           [val,idx] = max(corr);
           if i==1
              s(idx) = val; 
           end
           di = D(:,idx);
           nz_idx(i) = idx;
           D_temp(:,i) = di;
           if i>=2
              Dtemp_psuedo_inv = inv(transpose(D_temp)*D_temp)*transpose(D_temp);
              s_nz = Dtemp_psuedo_inv * x;
              s(nz_idx) = s_nz;
           end
           xr = xr - (transpose(xr)*di)*di;
           stopping_criteria = norm(x-D*s)/norm(x);
           i = i + 1;
           if i==N
              break 
           end
        end
        er = norm(x - D*s);
    end
end

function [er,s] = Basis_Pursuit(x,D)
    [~,N] = size(D);
    D_star = [D -D];
    f = (ones(2*N));
    f = f(:,1);
    lb = zeros(2*N,1);
    ub = +Inf(2*N,1);
    y = linprog(f,[],[],D_star,x,lb,ub);
    s = y(1:N) - y(N+1:end);
    er = norm(x - D * s);
end

function [er,s] = IRLS(x,D)
    [~,N] = size(D);
    a = 100;
    b = 1000;
    num_iter = 10;
    w = a*rand(N,1)+b;
    W = diag(w);
    for i=1:num_iter
        s = inv(W) * transpose(D) * (D*inv(W)*transpose(D))*x;
        w = zeros(N,1);
        for n=1:N
           if abs(s(n)) <= 5e-11
               w(n) = 1e6;
               s(n) = 0;
           else
               w(n) = 1/s(n);
           end
        end
        W = diag(w);
    end
    s = 4e10 * s;
    er = norm(x - D*s);
end
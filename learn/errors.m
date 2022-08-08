% This script collect the errors of the inverse results
model_path = './data/star3_kh10_n48_100/test';
num_results = 1;
results_dir = [model_path '/inverse/inverse'];
err_l2_all = zeros(1,num_results);
err_Chamfer_all = zeros(1,num_results);
for index=1:num_results
    temp = load([results_dir num2str(index) '.mat']);
    err_l2_all(index) = temp.err_l2;
    err_Chamfer_all(index) = temp.err_Chamfer(2);
end
err_l2_mean = mean(err_l2_all)
err_Chamfer_mean = mean(err_Chamfer_all)
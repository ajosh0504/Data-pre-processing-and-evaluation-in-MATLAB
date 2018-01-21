%The directory to be opened before running this is liblinear-2.1-->windows
addpath('../matlab');
clear all;
close all;
clc;
%Script to load data
raw_data_file = load('homework-6-data/covtype.data');
[m,n]=size(raw_data_file);
ind_train = load ('homework-6-data/covtype.train.index.txt');
ind_test = load ('homework-6-data/covtype.test.index.txt');
for i=1:m
    if(raw_data_file(i,n)== 2)
        raw_data_file(i,n)= 1;
    else
        raw_data_file(i,n)=-1;
    end
end
disp('hello');
%Split data into training and testing sets
testing_data =removerows(raw_data_file,'ind',ind_train);
training_data =removerows(raw_data_file,'ind', ind_test);
y_train=training_data(:,n);
x_train=training_data(:,1:(n-1));
validation_error=[];
C=[0.1 1 10 100 1000]; 
for i=1:5
model = train(y_train(104583:522910),sparse(x_train(104583:522910,:)),sprintf('-c %f -s 2',C(i)));
[~, v1, ~] = predict(y_train(1:104582),sparse(x_train(1:104582,:)), model);

set_2_x =removerows(x_train,'ind',[104583:209164]);
set_2_y =removerows(y_train,'ind',[104583:209164]);
model = train(set_2_y,sparse(set_2_x),sprintf('-c %f -s 2',C(i)));
[~, v2, ~] = predict(y_train(104583:209164),sparse(x_train(104583:209164,:)), model);

set_3_x =removerows(x_train,'ind',[209165:313746]);
set_3_y =removerows(y_train,'ind',[209165:313746]);
model = train(set_3_y,sparse(set_3_x),sprintf('-c %f -s 2',C(i)));
[~, v3, ~] = predict(y_train(209165:313746),sparse(x_train(209165:313746,:)), model);

set_4_x =removerows(x_train,'ind',[313747:418328]);
set_4_y =removerows(y_train,'ind',[313747:418328]);
model = train(set_4_y,sparse(set_4_x),sprintf('-c %f -s 2',C(i)));
[~, v4, ~] = predict(y_train(313747:418328),sparse(x_train(313747:418328,:)), model);

set_5_x =removerows(x_train,'ind',[418328:522910]);
set_5_y =removerows(y_train,'ind',[418328:522910]);
model = train(set_5_y,sparse(set_5_x),sprintf('-c %f -s 2',C(i)));
[~, v5, ~] = predict(y_train(418328:522910),sparse(x_train(418328:522910,:)), model);

s = v1(1)+v2(1)+v3(1)+v4(1)+v5(1);
%calculation average validation and training error for each C
validation_error = [validation_error (1-(s*0.002))];
end
[~,I_test] = min(validation_error);
y_testing=testing_data(:,end);
testing_data=testing_data(:,1:(end-1));
model = train(y_train,sparse(x_train),sprintf('-c %f -s 2',C(I_test)));
[predict_label, accuracy_test, ~] = predict(y_testing,sparse(testing_data), model);
perf_test_test=F1_score(y_testing,predict_label);
[X,Y,T,AUC_test] = perfcurve(y_testing,predict_label,1);

figure(1);
plot(X,Y);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curve');

%Re-scaling
rsc = raw_data_file;
min_mat=min(rsc);
max_mat=max(rsc);
for i= 1:(n-1)
    for j=1:m
        rsc(j,i)=(rsc(j,i)-min_mat(i))/(max_mat(i)-min_mat(i));
    end
end
testing_data =removerows(rsc,'ind',load ('homework-6-data/covtype.train.index.txt'));
training_data =removerows(rsc,'ind',load ('homework-6-data/covtype.test.index.txt'));
y_train=training_data(:,n);
x_train=training_data(:,1:(n-1));
validation_error=[];
C=[0.1 1 10 100 1000]; 
for i=1:5
model = train(y_train(104583:522910),sparse(x_train(104583:522910,:)),sprintf('-c %f -s 2',C(i)));
[~, v1, ~] = predict(y_train(1:104582),sparse(x_train(1:104582,:)), model);

set_2_x =removerows(x_train,'ind',[104583:209164]);
set_2_y =removerows(y_train,'ind',[104583:209164]);
model = train(set_2_y,sparse(set_2_x),sprintf('-c %f -s 2',C(i)));
[~, v2, ~] = predict(y_train(104583:209164),sparse(x_train(104583:209164,:)), model);

set_3_x =removerows(x_train,'ind',[209165:313746]);
set_3_y =removerows(y_train,'ind',[209165:313746]);
model = train(set_3_y,sparse(set_3_x),sprintf('-c %f -s 2',C(i)));
[~, v3, ~] = predict(y_train(209165:313746),sparse(x_train(209165:313746,:)), model);

set_4_x =removerows(x_train,'ind',[313747:418328]);
set_4_y =removerows(y_train,'ind',[313747:418328]);
model = train(set_4_y,sparse(set_4_x),sprintf('-c %f -s 2',C(i)));
[~, v4, ~] = predict(y_train(313747:418328),sparse(x_train(313747:418328,:)), model);

set_5_x =removerows(x_train,'ind',[418328:522910]);
set_5_y =removerows(y_train,'ind',[418328:522910]);
model = train(set_5_y,sparse(set_5_x),sprintf('-c %f -s 2',C(i)));
[~, v5, ~] = predict(y_train(418328:522910),sparse(x_train(418328:522910,:)), model);

s = v1(1)+v2(1)+v3(1)+v4(1)+v5(1);
%calculation average validation and training error for each C
validation_error = [validation_error (1-(s*0.002))];
end
[~,I_r] = min(validation_error);
y_testing=testing_data(:,end);
testing_data=testing_data(:,1:(end-1));
model = train(y_train,sparse(x_train),sprintf('-c %f -s 2',C(I_r)));
[predict_label, accuracy_test, ~] = predict(y_testing,sparse(testing_data), model);
perf_test_rescale=F1_score(y_testing,predict_label);
[X,Y,T,AUC_rescale] = perfcurve(y_testing,predict_label,1);

figure(2);
plot(X,Y);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curve - Rescaling');

%Standardization
stand = raw_data_file;
mean_mat=mean(stand);
std_mat=std(stand);
for i= 1:(n-1)
    for j=1:m
        stand(j,i)=(stand(j,i)-mean_mat(i))/(std_mat(i));
    end
end
%Split data into training and testing sets
testing_data =removerows(stand,'ind',load ('homework-6-data/covtype.train.index.txt'));
training_data =removerows(stand,'ind',load ('homework-6-data/covtype.test.index.txt'));
y_train=training_data(:,n);
x_train=training_data(:,1:(n-1));
validation_error=[];
C=[0.1 1 10 100 1000]; 
for i=1:5
model = train(y_train(104583:522910),sparse(x_train(104583:522910,:)),sprintf('-c %f -s 2',C(i)));
[~, v1, ~] = predict(y_train(1:104582),sparse(x_train(1:104582,:)), model);

set_2_x =removerows(x_train,'ind',[104583:209164]);
set_2_y =removerows(y_train,'ind',[104583:209164]);
model = train(set_2_y,sparse(set_2_x),sprintf('-c %f -s 2',C(i)));
[~, v2, ~] = predict(y_train(104583:209164),sparse(x_train(104583:209164,:)), model);

set_3_x =removerows(x_train,'ind',[209165:313746]);
set_3_y =removerows(y_train,'ind',[209165:313746]);
model = train(set_3_y,sparse(set_3_x),sprintf('-c %f -s 2',C(i)));
[~, v3, ~] = predict(y_train(209165:313746),sparse(x_train(209165:313746,:)), model);

set_4_x =removerows(x_train,'ind',[313747:418328]);
set_4_y =removerows(y_train,'ind',[313747:418328]);
model = train(set_4_y,sparse(set_4_x),sprintf('-c %f -s 2',C(i)));
[~, v4, ~] = predict(y_train(313747:418328),sparse(x_train(313747:418328,:)), model);

set_5_x =removerows(x_train,'ind',[418328:522910]);
set_5_y =removerows(y_train,'ind',[418328:522910]);
model = train(set_5_y,sparse(set_5_x),sprintf('-c %f -s 2',C(i)));
[~, v5, ~] = predict(y_train(418328:522910),sparse(x_train(418328:522910,:)), model);

s = v1(1)+v2(1)+v3(1)+v4(1)+v5(1);
%calculation average validation and training error for each C
validation_error = [validation_error (1-(s*0.002))];
end
[~,I_s] = min(validation_error);
y_testing=testing_data(:,end);
testing_data=testing_data(:,1:(end-1));
model = train(y_train,sparse(x_train),sprintf('-c %f -s 2',C(I_s)));
[predict_label, accuracy_test, ~] = predict(y_testing,sparse(testing_data), model);
perf_test_stand=F1_score(y_testing,predict_label);
[X,Y,T,AUC_stand] = perfcurve(y_testing,predict_label,1);

figure(3);
plot(X,Y);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curve - Standardization');

%Normalization
normal=raw_data_file;
norm_mat= sqrt(sum(normal.^2,2));
for i=1:(n-1)
    for j=1:m
        normal(j,i)=normal(j,i)/norm_mat(j);
    end
end
%Split data into training and testing sets
testing_data =removerows(stand,'ind',load ('homework-6-data/covtype.train.index.txt'));
training_data =removerows(stand,'ind',load ('homework-6-data/covtype.test.index.txt'));
y_train=training_data(:,n);
x_train=training_data(:,1:(n-1));
validation_error=[];
C=[0.1 1 10 100 1000]; 
for i=1:5
model = train(y_train(104583:522910),sparse(x_train(104583:522910,:)),sprintf('-c %f -s 2',C(i)));
[~, v1, ~] = predict(y_train(1:104582),sparse(x_train(1:104582,:)), model);

set_2_x =removerows(x_train,'ind',[104583:209164]);
set_2_y =removerows(y_train,'ind',[104583:209164]);
model = train(set_2_y,sparse(set_2_x),sprintf('-c %f -s 2',C(i)));
[~, v2, ~] = predict(y_train(104583:209164),sparse(x_train(104583:209164,:)), model);

set_3_x =removerows(x_train,'ind',[209165:313746]);
set_3_y =removerows(y_train,'ind',[209165:313746]);
model = train(set_3_y,sparse(set_3_x),sprintf('-c %f -s 2',C(i)));
[~, v3, ~] = predict(y_train(209165:313746),sparse(x_train(209165:313746,:)), model);

set_4_x =removerows(x_train,'ind',[313747:418328]);
set_4_y =removerows(y_train,'ind',[313747:418328]);
model = train(set_4_y,sparse(set_4_x),sprintf('-c %f -s 2',C(i)));
[~, v4, ~] = predict(y_train(313747:418328),sparse(x_train(313747:418328,:)), model);

set_5_x =removerows(x_train,'ind',[418328:522910]);
set_5_y =removerows(y_train,'ind',[418328:522910]);
model = train(set_5_y,sparse(set_5_x),sprintf('-c %f -s 2',C(i)));
[~, v5, ~] = predict(y_train(418328:522910),sparse(x_train(418328:522910,:)), model);

s = v1(1)+v2(1)+v3(1)+v4(1)+v5(1);
%calculation average validation and training error for each C
validation_error = [validation_error (1-(s*0.002))];
end
[~,I_n] = min(validation_error);
y_testing=testing_data(:,end);
testing_data=testing_data(:,1:(end-1));
model = train(y_train,sparse(x_train),sprintf('-c %f -s 2',C(I_n)));
[predict_label, accuracy_test, ~] = predict(y_testing,sparse(testing_data), model);
perf_test_norm=F1_score(y_testing,predict_label);
[X,Y,T,AUC_norm] = perfcurve(y_testing,predict_label,1);

figure(4);
plot(X,Y);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curve - Normalization');


% Part 2:
best_preprocess = 0;
if (AUC_rescale > AUC_test && AUC_rescale > AUC_stand && AUC_rescale > AUC_norm)
	best_preprocess = 2;
end
if (AUC_stand > AUC_test && AUC_rescale < AUC_stand && AUC_stand > AUC_norm)
	best_preprocess = 3;
end
if (AUC_norm > AUC_test && AUC_rescale < AUC_norm && AUC_stand < AUC_norm)
	best_preprocess = 4;
end
raw_data_file = load('homework-6-data/covtype.data');[m,n]=size(raw_data_file);
for i=1:m
    if(raw_data_file(i,n)== 5)
        raw_data_file(i,n)= 1;
    else
        raw_data_file(i,n)=-1;
    end
end


if (best_preprocess == 2)
	rsc = raw_data_file;
	min_mat=min(rsc);
	max_mat=max(rsc);
	for i= 1:(n-1)
		for j=1:m
			rsc(j,i)=(rsc(j,i)-min_mat(i))/(max_mat(i)-min_mat(i));
		end
	end
	raw_data_file = rsc;
end
if (best_preprocess == 3)
	stand = raw_data_file;
	mean_mat=mean(stand);
	std_mat=std(stand);
	for i= 1:(n-1)
		for j=1:m
			stand(j,i)=(stand(j,i)-mean_mat(i))/(std_mat(i));
		end
	end
	raw_data_file = stand;
end
if (best_preprocess == 4)
	normal=raw_data_file;
	norm_mat= sqrt(sum(normal.^2,2));
	for i=1:(n-1)
		for j=1:m
			normal(j,i)=normal(j,i)/norm_mat(j);
		end
	end
	raw_data_file = stand;
end

testing_data =removerows(raw_data_file,'ind',load ('homework-6-data/covtype.train.index.txt'));
training_data =removerows(raw_data_file,'ind',load ('homework-6-data/covtype.test.index.txt'));
y_train=training_data(:,n);
x_train=training_data(:,1:(n-1));
validation_error=[];
training_error=[];
C=[0.1 1 10 100 1000]; 
for i=1:5
model = train(y_train(104583:522910),sparse(x_train(104583:522910,:)),sprintf('-c %f -s 2',C(i)));
[~, v1, ~] = predict(y_train(1:104582),sparse(x_train(1:104582,:)), model);

set_2_x =removerows(x_train,'ind',[104583:209164]);
set_2_y =removerows(y_train,'ind',[104583:209164]);
model = train(set_2_y,sparse(set_2_x),sprintf('-c %f -s 2',C(i)));
[~, v2, ~] = predict(y_train(104583:209164),sparse(x_train(104583:209164,:)), model);

set_3_x =removerows(x_train,'ind',[209165:313746]);
set_3_y =removerows(y_train,'ind',[209165:313746]);
model = train(set_3_y,sparse(set_3_x),sprintf('-c %f -s 2',C(i)));
[~, v3, ~] = predict(y_train(209165:313746),sparse(x_train(209165:313746,:)), model);

set_4_x =removerows(x_train,'ind',[313747:418328]);
set_4_y =removerows(y_train,'ind',[313747:418328]);
model = train(set_4_y,sparse(set_4_x),sprintf('-c %f -s 2',C(i)));
[~, v4, ~] = predict(y_train(313747:418328),sparse(x_train(313747:418328,:)), model);

set_5_x =removerows(x_train,'ind',[418328:522910]);
set_5_y =removerows(y_train,'ind',[418328:522910]);
model = train(set_5_y,sparse(set_5_x),sprintf('-c %f -s 2',C(i)));
[~, v5, ~] = predict(y_train(418328:522910),sparse(x_train(418328:522910,:)), model);

s = v1(1)+v2(1)+v3(1)+v4(1)+v5(1);
%calculation average validation and training error for each C
validation_error = [validation_error (1-(s*0.002))];
end

%Best C is the one with least validation error
[~,I] = min(validation_error);
y_testing=testing_data(:,end);
testing_data=testing_data(:,1:(end-1));
model = train(y_train,sparse(x_train),sprintf('-c %f -s 2',C(I)));
[predict_label, accuracy_test, ~] = predict(y_testing,sparse(testing_data), model);
perf=F1_score(y_testing,predict_label);
[X,Y,T,AUC_test] = perfcurve(y_testing,predict_label,1);

figure(5);
plot(X,Y);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC Curve - Part 2');






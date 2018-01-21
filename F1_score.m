function [ score ] = F1_score( y1, y2 )
%F1_SCORE Summary of this function goes here
%   Detailed explanation goes here
TN = 0;TP = 0;FN = 0;FP = 0;
[m,~] = size(y1);
for i = 1:m
    if y1(i) == y2(i)
        if y1(i) == 1
            TP = TP + 1;
        else
            TN = TN + 1;
        end
    else
        if y1(i) == 1
            FP = FP + 1;
        else
            FN = FN + 1;
        end
    end
end

a = TP / (TP + FP);
b = TP / (TP + FN);

score = 2*a*b/(a+b);

end


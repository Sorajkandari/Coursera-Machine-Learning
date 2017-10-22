function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

% we are running 1000 loops with each loop of size stepsize

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval) % here the for loop start with min(pval) end with max(pval)
    % and the increment of stepsize in each step.
  
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

   prediction=(pval<epsilon);
   TruePos=sum((prediction==1)&(yval==1));
   
   FalsePos=sum((prediction==1)&(yval==0));
   
   prescise=TruePos/(TruePos+FalsePos);
   
   FalseNeg=sum((prediction==0)&(yval==1));
   
   recall=TruePos/(TruePos+FalseNeg);
   
   F1=(2*prescise*recall)/(prescise+recall);
   
   %actuall F1 gives the best scores measn more the F1 value more will be our assumption correct.

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end

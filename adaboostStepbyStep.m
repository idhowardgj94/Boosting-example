%run example from http://www.csie.ntu.edu.tw/~b92109/course/Machine%20Learning/AdaBoostExample.pdf

%adaboost重要的參數：error rate, alpha, q, pro.
clear;
clc;
x=[0 1 2 3 4 5 6 7 8 9];
y=[1 1 1 -1 -1 -1 1 1 1 -1];
trainSet=[x;y];
trainSetBlock=2;
probability=zeros(10,10);
probability(1,:)=linspace(0.1, 0.1, 10);
errorRate=zeros(1,10);
correctInstance=0;
wrongInstance=0;
threshold=0.5:1:8.5;
alpha=zeros(1,10);
localMinRate=9999;
answer=zeros(1,10);
localAnswer=zeros(1,10);
Hfunction=zeros(1,10);
chooseThreshold=zeros(2,10);
for loop=1:3
    %trainIndex=(loop-1)*trainSetBlock;
    localMinRate=9999;
    %% coculate error rate.
    for thresholdIndex=1:size(threshold,2)
        
        sign=1;
        s=1;
        for sign=1:2
            correctInstance=0;
            wrongInstance=0;
            if sign==2
                s=s*-1;
            end
            for i=1:size(trainSet, 2)
                if(s*trainSet(1, i)<s*threshold(thresholdIndex) && trainSet(2, i)==1)
                    correctInstance=correctInstance+1*probability(loop, i);
                    localAnswer(i)=1;
                elseif(s*trainSet(1, i)<s*threshold(thresholdIndex) && trainSet(2, i)==-1)
                    wrongInstance=wrongInstance+1*probability(loop, i);
                    localAnswer(i)=1;
                elseif(s*trainSet(1, i)>=s*threshold(thresholdIndex)&&trainSet(2, i)==1)
                    wrongInstance=wrongInstance+1*probability(loop, i);
                    localAnswer(i)=-1;
                elseif(s*trainSet(1,i)>=s*threshold(thresholdIndex)&&trainSet(2, i)==-1)
                    correctInstance=correctInstance+1*probability(loop, i);
                    localAnswer(i)=-1;
                end
            end
            localErrorRate=wrongInstance;
            if(localErrorRate<localMinRate)
                localMinRate=localErrorRate;
                answer=localAnswer;
                chooseThreshold(1, loop)=threshold(thresholdIndex);
                chooseThreshold(2, loop)=sign;
            end
        end
        
    end
    fprintf(['\n\nround', num2str(loop), ', choose threshold=', num2str(chooseThreshold(1,loop))]);
    errorRate(loop)=localMinRate;
    fprintf(['\nerror rate: ', num2str(errorRate(loop)), '\n']);
    alpha(loop)=log((1-errorRate(loop))/errorRate(loop))*0.5;
    %qWrong=exp(alpha1*1);
    %qRight=exp(-alpha1*1);
    
    %% compte new probability
    probability(loop+1,:)=probability(loop, :).*(exp(-1*alpha(loop)*trainSet(2, :).*answer));
    Z=sum(probability(loop+1, :));
    probability(loop+1,:)=probability(loop+1,:)/Z;
    fprintf(['new probability: \n', num2str(probability(loop+1, :))]);
    Hfunction(loop)=alpha(loop);
    fprintf(['\nf',num2str(loop),'(x) = '])
    for looptemp=1:loop-1
        if (chooseThreshold(2, looptemp)==1)
            fprintf([num2str(Hfunction(looptemp)),' I(X<',num2str(chooseThreshold(1, looptemp)), ') + ']);
        elseif(chooseThreshold(2, looptemp)==2)
            fprintf([num2str(Hfunction(looptemp)),' I(X>',num2str(chooseThreshold(1, looptemp)), ') + ']);
        end
    end
    if (chooseThreshold(2, loop)==1)
        fprintf([num2str(Hfunction(loop)),' I(X<',num2str(chooseThreshold(1, loop)), ')\n']);
    elseif(chooseThreshold(2, loop)==2)
        fprintf([num2str(Hfunction(loop)),' I(X>',num2str(chooseThreshold(1, loop)), ')\n']);
    end
    
    %% do a predict with new boost model
    correctInstance=0;
    wrongInstance=0;
    finalPredictHypothsis=zeros(size(Hfunction, 2), size(trainSet, 2));
    finalPredictAnswer=zeros(1, 10);
    for pIndex=1:size(trainSet, 2)
        for Hindex=1:size(find(Hfunction(1,:)~=0), 2)
            if((trainSet(1,pIndex)<chooseThreshold(1,Hindex)&&chooseThreshold(2,Hindex)==1)||(trainSet(1,pIndex)>chooseThreshold(1,Hindex)&&chooseThreshold(2,Hindex)==2))
                finalPredictHypothsis(Hindex, pIndex)=1;
            elseif((trainSet(1,pIndex)<chooseThreshold(1,Hindex)&&chooseThreshold(2,Hindex)==2)||(trainSet(1,pIndex)>chooseThreshold(1,Hindex)&&chooseThreshold(2,Hindex)==1))
                finalPredictHypothsis(Hindex, pIndex)=-1;
            end
        end
        oneIndex=find(finalPredictHypothsis(:,pIndex)==1);
        minusoneIndex=find(finalPredictHypothsis(:,pIndex)==-1);
        groupOneVote=finalPredictHypothsis(oneIndex, pIndex)'*Hfunction(oneIndex)';
        groupMinusVote=abs(finalPredictHypothsis(minusoneIndex, pIndex)'*Hfunction(minusoneIndex)');
        if(isempty(groupMinusVote))
            groupMinusVote=0;
        elseif(isempty(groupOneVote))
            groupOneVote=0;
        end
        if(groupOneVote>groupMinusVote && trainSet(2, pIndex)==1)
            correctInstance=correctInstance+1;
            finalPredictAnswer(pIndex)=1;
        elseif(groupOneVote>groupMinusVote && trainSet(2, pIndex)==-1)
            wrongInstance=wrongInstance+1;
            finalPredictAnswer(pIndex)=1;
        elseif(groupOneVote<groupMinusVote &&trainSet(2, pIndex)==1)
            wrongInstance=wrongInstance+1;
            finalPredictAnswer(pIndex)=-1;
        elseif(groupOneVote<groupMinusVote&&trainSet(2, pIndex)==-1)
            correctInstance=correctInstance+1;
            finalPredictAnswer(pIndex)=-1;
        end
    end
    fprintf(['predict result: ', num2str(wrongInstance), ' mistake\n', num2str(finalPredictAnswer),'\n']);
    
end
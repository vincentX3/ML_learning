# partial codes in week_2 code assignment.
>written by *VincentX3*, Nov.08.18
### in plotData.m

```matlab
plot(x,y,'rx','MarkerSize',10);
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');
%the first data visualizing function I meet in matlab.
%use legend() to add "legend"
```
### in computeCost.m

```matlab
J=sum((X*theta-y).^2)/(2*m);
```
>form a habit to consider every variate is **vector**.  
this would help us reuse code in multivariate situation.
### in gradientDescent.m

```matlab
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%by save the history we can plot the process of convergence.
%also help us to check whether we code properly.
%if J_history go up it mean something wrong in our codes.
for iter = 1:num_iters
    theta=theta-(alpha/m).*X'*(X*theta-y);
	%suggest that: derive the equation by hand before code.
	%make sure u clearly know the size of matrix u manipulating.
	J_history(iter) = computeCost(X, y, theta);
end
```
>	suggest that: derive the equation **by hand** before code.  
	make sure u clearly know the **size** of matrix u manipulating.

### in featureNormalize.m
```matlab
mu=mean(X);
sigma=std(X);
X_norm=(X_norm-mu)./sigma;    
```

### optional:Estimate the price of a 1650 sq-ft, 3 br house
```matlab
new=[1650,3];
new_normalize=(new-mu)./sigma;
new_normalize=[ones(1,1),new_normalize];
price = new_normalize*theta; % You should change this
```

### get a intuition of our work
```matlab
%ploting the linear function with data set, 
%to see how close the answer do we get.
y_predict=X*theta;
plot(data(:,1),data(:,3),'o',data(:,1),y_predict,'*');
ylabel("house price");
xlabel("size of the house");
legend("data","prediction");

%picture see week2_code_pic1.jpg
```
![picture see week2_code_pic1.png](/week2_code_pic1.png)

### in normalEQN.m
```matlab
theta=(X'*X)\X'*y;%by matlab's suggestion, use'\' instead of inv() .
```


### more details about code
>why save mean and std values

>for future prediction, which we still need to normalize the input, need the same values we used in training set.
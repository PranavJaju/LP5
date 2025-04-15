def knapsack(C,n): 

    if n<0 or C<=0:         #base case
        return 0
        
    if wt[n]>C:             #Higher weight than available
        return knapsack(C, n-1)

    else:
        return max(val[n] + knapsack(C-wt[n],n-1),knapsack(C,n-1))      # max(including , not including)

wt=[0, 2, 2, 4, 5] # Weight array
val=[0, 3, 7, 2, 9] #value array
C=10
n=len(val) - 1        
print("Max Value is: ",knapsack(C,n))

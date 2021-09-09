/*
An algorithm which solves the problem described in "Tiling by Squares":https://www.codingame.com/training/expert/tiling-by-squares
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define L 100

int solve(int,int,int*,int);
int gcd(int,int);
int best_sol=0;

int main(){
    int w,h;
    printf("Please enter width and height of the rectangle (maximum dimensions are 99x99)\n");
    scanf("%d%d",&w,&h);
    int d=gcd(w,h);
    w/=d;
    h/=d;
    int a=(w<h)?w:h;
    int b=(w>h)?w:h;
    int c;

    while(a>1){
        best_sol+=1;
        b-=a;
        c=(a<b)?a:b;
        b=a+b-c;
        a=c;
    }
    best_sol+=a*b;

    int profile[L];

    for(int i=0;i<w;i++){
        profile[i]=0;
    }

    printf("The minimal square tiling is: %d.\n",solve(w,h,profile,0));
}

int solve(int w,int h,int*profile,int running_total){
    if(running_total>best_sol)return w*h;
    int min_h=h;
    int left=0;
    for(int i=0;i<w;i++){
        if(profile[i]<min_h){
            min_h=profile[i];
            left=i;
        }
    }
    if(min_h==h)return 0;
    int right=left+1;
    while(((right<w)?(profile[right]==min_h):false)&&(min_h+right-left<h))++right;
    int sol=w*h;
    for(int i=right;i>left;--i){
        int new_profile[L];
        for(int j=0;j<w;j++){
            new_profile[j]=profile[j];
        }
        for(int j=left;j<i;j++){
            new_profile[j]=min_h+i-left;
        }
        int try=1+solve(w,h,new_profile,running_total+1);
        if(try<sol)sol=try;
        if(sol+running_total<best_sol)best_sol=sol+running_total;
    }
    return sol;
}

int gcd(int a,int b){
if(a*b==0)return a+b;
int c=a<b?a:b;
b+=a-c;
a=c;
return gcd(a,b%a);
}

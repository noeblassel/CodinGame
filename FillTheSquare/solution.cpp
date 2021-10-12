#include <iostream>
#include <vector>
#include <algorithm>

/*
Solution to the problem "Fill the Square" on CodinGame
https://www.codingame.com/training/expert/fill-the-square

Statement:

James Bond needs to infiltrate the SPECTRE headquarters. The door lock looks like a square of N×N LEDs.
Some of those LEDs are lit, and each LED’s state can be changed by touching it.
Unfortunately, when you touch an LED, not only does its state change, but the states of its horizontal and vertical neighbors change as well.

So for an input of

...
.*.
...

If Bond were to press

...
.X.
...

Then the next state would be

.*.
*.*
.*.

Bond’s only option is to light all the LEDs to unlock the door…
You are Q, and 007 contacts you to request your help. Try to find a way to quickly solve the puzzle (with the minimum number of touches) and get your agent into the house!
*/

using namespace std;
typedef vector<vector<bool>> m2mat;//A typedef for a mod 2 matrix
vector<bool> lin_solve_mod2(m2mat A,vector<bool> b,int N);

/*Every initially lit/non-lit LED must have an even/odd number of lit neighbors in the final state.
The problem can be represented as a linear system mod 2 with N*N unknowns, Ax=b, where A is the adjacency matrix
describing neighborhood relationships between points on the grid, and b describes the parity requirements for each point on the grid.*/

int main()
{
    //Read constraints
    int N;
    cin >> N; cin.ignore();
    vector<bool>b;

    for (int i = 0; i < N; i++) {
        string ROW;
        getline(cin, ROW);
        for(auto& c:ROW)b.push_back(c=='.');
    }

    //Compute adjacency matrix A
    m2mat A;
    for(int j=0;j<N*N;j++){
        vector<bool>col{};
        for(int i=0;i<N*N;i++){
            int dx=(i%N)-(j%N);
            int dy=(i/N)-(j/N);
            col.push_back((dx<=1)&&(dx>=-1)&&(dy<=1)&&(dy>=-1)&&(dx*dy==0));//description of neighborhood--note every point is a self-neighbor
        }
        A.push_back(col);
    }
    //Solve linear system
    vector<bool> sol=lin_solve_mod2(A,b,N*N);

    //Output solution
    for(int i=0;i<N*N;i++){
        cout<<(sol[i]?'X':'.');
        if((i+1)%N==0)cout<<endl;
    }

    return 0;
}

vector<bool>lin_solve_mod2(m2mat A,vector<bool>b,int N){
    //Solve the system by Gaussian elimination in Z/2Z

    int x_pivot=0;
    int y_pivot=0;

    while((x_pivot<N)&&(y_pivot<N)){
        int next_y_pivot=y_pivot;
        while(!A[x_pivot][next_y_pivot]){
            next_y_pivot++;
            if(next_y_pivot==N)break;
        }
        if(next_y_pivot==N)x_pivot++;
        else{
            swap(A[y_pivot],A[next_y_pivot]);
            swap(b[y_pivot],b[next_y_pivot]);//Don't forget we are operating on an augmented matrix

            for(int row=y_pivot+1;row<N;row++){
                if(A[row][x_pivot]){
                    A[row][x_pivot]=0;
                    for(int col=x_pivot+1;col<N;col++)A[row][col]=A[row][col]^A[y_pivot][col];
                    b[row]=b[row]^b[y_pivot];//Augmented row operations!
                }
            }
            x_pivot++;
            y_pivot++;
        }
    }
    //matrix is now in row-echelon form, we can now solve by substitution
    vector<bool>sol;
    for(int j=0;j<N;j++){
        bool x=b[N-j-1];
        for(int i=0;i<j;i++){
            x^=(A[N-j-1][N-i-1]&sol[i]);
        }
        sol.push_back(x);
    }
    reverse(sol.begin(),sol.end());
    return sol;
}

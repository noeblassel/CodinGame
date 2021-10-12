#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
/*
Solution to the problem "Doubly Solved Rubik's Cube" on CodinGame.com
https://www.codingame.com/training/hard/doubly-solved-rubiks-cube

Statement:
Given a configuration of the Rubik's cube, find its doubly solved configuration. Specifically, pick any sequence of twists that solves the given Rubik's cube. Apply that same sequence twice to the given cube (once to solve it, then again to get some new configuration). You are asked to output the resulting configuration.

It can be shown that the result is independent of the sequence of twists used to solve the initial cube. In this problem only the configuration below counts as solved (no other orientations).

The solved cube configuration is:

    UUU
    UUU
    UUU

LLL FFF RRR BBB
LLL FFF RRR BBB
LLL FFF RRR BBB

    DDD
    DDD
    DDD

There are no trailing spaces.

*/
using namespace std;

int main()
{
    //Parse input
    string row;
    string input_cube;
    
    for(int i=0;i<18;++i){
        cin>>row;
        input_cube+=row;
    }
    array<int,54> permutation;
    array<int,54> inverse_permutation;

    /* FACE MAPPING:

            00 01 02
            03 04 05
            06 07 08

09 10 11    12 13 14    15 16 17    18 19 20
21 22 23    24 25 26    27 28 29    30 31 32
33 34 35    36 37 38    39 40 41    42 43 44

            45 46 47
            48 49 50
            51 52 53

    */
    //Define a map to assign to each face a unique face id. This encodes adjacency relationships between faces.
    map<set<char>,map<char,int>> face_identificator;

    //Corner cubes
    face_identificator[set<char>{'U','B','L'}]=map<char,int>{{'U',0},{'B',20},{'L',9}};
    face_identificator[set<char>{'U','B','R'}]=map<char,int>{{'U',2},{'B',18},{'R',17}};
    face_identificator[set<char>{'U','F','L'}]=map<char,int>{{'U',6},{'F',12},{'L',11}};
    face_identificator[set<char>{'U','F','R'}]=map<char,int>{{'U',8},{'F',14},{'R',15}};
    face_identificator[set<char>{'D','B','L'}]=map<char,int>{{'D',51},{'B',44},{'L',33}};
    face_identificator[set<char>{'D','F','L'}]=map<char,int>{{'D',45},{'F',36},{'L',35}};
    face_identificator[set<char>{'D','F','R'}]=map<char,int>{{'D',47},{'F',38},{'R',39}};
    face_identificator[set<char>{'D','B','R'}]=map<char,int>{{'D',53},{'B',42},{'R',41}};

    //Two-faced cubes
    face_identificator[set<char>{'U','B'}]=map<char,int>{{'U',1},{'B',19}};
    face_identificator[set<char>{'U','L'}]=map<char,int>{{'U',3},{'L',10}};
    face_identificator[set<char>{'U','R'}]=map<char,int>{{'U',5},{'R',16}};
    face_identificator[set<char>{'U','F'}]=map<char,int>{{'U',7},{'F',13}};
    face_identificator[set<char>{'L','B'}]=map<char,int>{{'L',21},{'B',32}};
    face_identificator[set<char>{'L','F'}]=map<char,int>{{'L',23},{'F',24}};
    face_identificator[set<char>{'F','R'}]=map<char,int>{{'F',26},{'R',27}};
    face_identificator[set<char>{'R','B'}]=map<char,int>{{'R',29},{'B',30}};
    face_identificator[set<char>{'D','F'}]=map<char,int>{{'F',37},{'D',46}};
    face_identificator[set<char>{'L','D'}]=map<char,int>{{'L',34},{'D',48}};
    face_identificator[set<char>{'D','R'}]=map<char,int>{{'D',50},{'R',40}};
    face_identificator[set<char>{'D','B'}]=map<char,int>{{'D',52},{'B',43}};

    //Center cubes
    face_identificator[set<char>{'U'}]=map<char,int>{{'U',4}};
    face_identificator[set<char>{'L'}]=map<char,int>{{'L',22}};
    face_identificator[set<char>{'F'}]=map<char,int>{{'F',25}};
    face_identificator[set<char>{'R'}]=map<char,int>{{'R',28}};
    face_identificator[set<char>{'B'}]=map<char,int>{{'B',31}};
    face_identificator[set<char>{'D'}]=map<char,int>{{'D',49}};

    //Determine the permutation of faces in the input cube-- e.g if the first inputed label is 'F', it can theoretically be one of the 'F' faces of the four corner cubes with 'F' faces on them.
    vector<array<int,3>>corner_cubes{{0,20,9},{2,17,18},{6,11,12},{8,14,15},{33,44,51},{35,36,45},{38,39,47},{41,42,53}};
    vector<array<int,2>>two_face_cubes{{1,19},{3,10},{5,16},{7,13},{21,32},{23,24},{26,27},{29,30},{37,46},{34,48},{50,40},{52,43}};
    vector<int>center_cubes{4,22,25,28,31,49};
    //compute the permutation
    int i,j,k;
    char li,lj,lk;
    for(auto& cube:corner_cubes){
        i=cube[0],j=cube[1],k=cube[2];
        li=input_cube[i],lj=input_cube[j],lk=input_cube[k];
        set<char>S{li,lj,lk};
        permutation[i]=face_identificator[S][li];
        permutation[j]=face_identificator[S][lj];
        permutation[k]=face_identificator[S][lk];
    }
    for(auto& cube:two_face_cubes){
        i=cube[0],j=cube[1];
        li=input_cube[i],lj=input_cube[j];
        set<char>S{li,lj};
        permutation[i]=face_identificator[S][li];
        permutation[j]=face_identificator[S][lj];
    }
    for(auto& i:center_cubes){
        li=input_cube[i];
        set<char>S{li};
        permutation[i]=face_identificator[S][li];
    } 
    //Compute the inverse of this permutation--this is the result of reversing the moves
    for(int i=0;i<54;i++){
        j=0;
        while(permutation[j]!=i)j++;
        inverse_permutation[i]=j;
    }
    //Output this inverse permutation
    string label_str{"UUUUUUUUULLLFFFRRRBBBLLLFFFRRRBBBLLLFFFRRRBBBDDDDDDDDD"};
    string output;
    for(int i=0;i<54;i++)output+=label_str[inverse_permutation[i]];

    cout<<"    "<<output.substr(0,3)<<endl;
    cout<<"    "<<output.substr(3,3)<<endl;
    cout<<"    "<<output.substr(6,3)<<endl<<endl;

    cout<<output.substr(9,3)<<" "<<output.substr(12,3)<<" "<<output.substr(15,3)<<" "<<output.substr(18,3)<<endl;
    cout<<output.substr(21,3)<<" "<<output.substr(24,3)<<" "<<output.substr(27,3)<<" "<<output.substr(30,3)<<endl;
    cout<<output.substr(33,3)<<" "<<output.substr(36,3)<<" "<<output.substr(39,3)<<" "<<output.substr(42,3)<<endl<<endl;

    cout<<"    "<<output.substr(45,3)<<endl;
    cout<<"    "<<output.substr(48,3)<<endl;
    cout<<"    "<<output.substr(51,3)<<endl;

    return 0;
}

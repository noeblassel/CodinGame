#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/*
Solution to the problem "Fractal Carpet" on CodinGame.com
https://www.codingame.com/training/hard/fractal-carpet

Statement:
Aladdin was out celebrating last night and smashed his magic carpet into a spire.
He managed to escape injury but he somehow managed to tear a chunk out of the fabric and can't find the missing piece.
Unfortunately the cost of the magic thread used in the carpet is extremely expensive even for a prince so he'll have to patch the hole by providing the missing pattern.
This wouldn't be such a big deal but due to the nature of magic the carpet has a very peculiar fractal pattern to it.
As Aladdin inspects the whole carpet at various levels of detail this is what he would normally see:

Level 0:
0

Level 1:
000
0+0
000

Level 2:
000000000
0+00+00+0
000000000
000+++000
0+0+++0+0
000+++000
000000000
0+00+00+0
000000000

Level 3:
000000000000000000000000000
0+00+00+00+00+00+00+00+00+0
000000000000000000000000000
000+++000000+++000000+++000
0+0+++0+00+0+++0+00+0+++0+0
000+++000000+++000000+++000
000000000000000000000000000
0+00+00+00+00+00+00+00+00+0
000000000000000000000000000
000000000+++++++++000000000
0+00+00+0+++++++++0+00+00+0
000000000+++++++++000000000
000+++000+++++++++000+++000
0+0+++0+0+++++++++0+0+++0+0
000+++000+++++++++000+++000
000000000+++++++++000000000
0+00+00+0+++++++++0+00+00+0
000000000+++++++++000000000
000000000000000000000000000
0+00+00+00+00+00+00+00+00+0
000000000000000000000000000
000+++000000+++000000+++000
0+0+++0+00+0+++0+00+0+++0+0
000+++000000+++000000+++000
000000000000000000000000000
0+00+00+00+00+00+00+00+00+0
000000000000000000000000000

etc.

The seamstress will give you the level of detail that she needs and the top left and bottom right coordinates of the piece she needs the pattern for.
*/

typedef unsigned long ulong;
int main()
{
    int L;
    scanf("%d", &L);
    ulong x1,y1,x2,y2;
    scanf("%ld%ld%ld%ld", &x1, &y1, &x2, &y2);
    for(ulong j=y1;j<=y2;j++){
        for(ulong i=x1;i<=x2;i++){
            ulong a=i;
            ulong b=j;
            bool in=true;
            while((a!=0)||(b!=0)){
                if((a%3==1)&&(b%3==1)){
                    in=false;
                    break;
                }
                a/=3;
                b/=3;
            }
            printf("%c",in?'0':'+');
        }
        printf("\n");
    }

    return 0;
}

#include <bits/stdc++.h>

using namespace std;

int main(){
    int T;
    int C = 1;
    cin >> T;
    while (T--){
        long long a[3];
        cin >> a[0] >> a[1] >> a[2];
        bool flag = 0;
        for (int i = 0; i < 3; i++){
            if (flag){
                break;
            }
            for (int j = 0; j < 3; j++){
                if (flag){
                    break;
                }
                if (i == j){
                    continue;
                }
                for (int k = 0; k < 3; k++){
                    if (flag){
                        break;
                    }
                    if (j == k || k == i){
                        continue;
                    }
                    long long seconds = (a[j]%720000000000)*60;
                    double diff = (seconds - a[i])/59.;
                    long long minute = (a[k]%3600000000000)*12;
                    double diff2 = (minute - a[j])/11.;
                    if (diff == diff2){
                        diff *= 720;
                        int hour = (a[k]+diff)/3600000000000;
                        int min = (a[j]+diff)/720000000000;
                        int sec = (a[i]+diff)/720000000000;
                        int nano = (a[i]+diff)%720000000000;
                        printf("Case #%d: %d %d %d %d\n", C++, hour, min, sec, nano);
                        flag = 1;
                        break;
                    }

                }
            }
        }
        
    }
    return 0;
}
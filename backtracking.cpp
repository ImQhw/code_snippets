#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>

using namespace std;

using v_int = vector<int>;
using v_bool = vector<bool>;
using vv_int = vector<v_int>;

void print(const vv_int &vv)
{
    cout << "print vv with " << vv.size() << " entries" << endl;

    for (const auto &v : vv)
    {
        for (const auto &a : v)
        {
            cout << a << ", ";
        }
        cout << endl;
    }
    cout << endl;
}

void combination(vv_int &ans, v_int &track, int n, int k, int startIndex)
{
    if (track.size() == k)
    {
        ans.emplace_back(track);
        return;
    }

    for (int i = startIndex; i <= n; i++)
    {
        track.push_back(i);
        combination(ans, track, n, k, i + 1);
        track.pop_back();
    }
}

void combination(vv_int &ans, int n, int k)
{
    v_int track;
    ans.clear();

    combination(ans, track, n, k, 1);
}

void permutation(vv_int &ans, v_int &track, v_bool &used, int n, int k)
{
    if (track.size() == k)
    {
        ans.push_back(track);
        return;
    }

    for (int i = 1; i <= n; i++)
    {
        if (!used[i])
        {
            used[i] = true;
            track.push_back(i);
            permutation(ans, track, used, n, k);
            used[i] = false;
            track.pop_back();
        }
    }
}

void permutation(vv_int &ans, int n, int k)
{
    v_int track;
    v_bool used(n, false);
    ans.clear();

    permutation(ans, track, used, n, k);
}

int main()
{
    vv_int ans;

    combination(ans, 10, 2);
    print(ans);

    combination(ans, 10, 3);
    print(ans);

    permutation(ans, 10, 2);
    print(ans);
}

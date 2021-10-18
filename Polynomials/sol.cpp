#include <bitset>
#include <algorithm>
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <map>

using namespace std;

#define __NODEBUG__

//A suboptimal implementation of Berlekamp's algorithm over F2 to factor binary polynomials into irreducible factors


using F2vec = vector<bool>;
using F2mat = vector<F2vec>;

//bit-level ops
F2vec from_ints(vector<unsigned int> v);
vector<unsigned int> to_ints(F2vec P);
F2vec operator>>(F2vec P, int n);
F2vec operator<<(F2vec P, int n);
F2vec operator^(F2vec P, F2vec Q);
void trim(F2vec &P); //resizes P to fit its actual non-zero coefficients
void untrim(F2vec &P, int d);

//polynomial arithmetic
pair<F2vec, F2vec> divmod(F2vec P, F2vec Q);
F2vec operator*(F2vec P, F2vec Q);
F2vec operator/(F2vec P, F2vec Q) { return divmod(P, Q).first; }
F2vec operator%(F2vec P, F2vec Q);
F2vec gcd(F2vec P, F2vec Q);

map<F2vec, vector<F2vec>> factorization_memo;
vector<F2vec> factor_poly(F2vec P);

//linear algebra
pair<F2mat, F2mat> sq_row_reduced(F2mat A);
vector<F2vec> sq_kernel(F2mat A);

//debug utilities
#ifdef __DEBUG__
void print_poly(F2vec P, size_t size, ostream &str);
void print_sq_mat(F2mat M);
void print_sq_reduction_pair(F2mat M);
void print_sq_mat_pair(pair<F2mat, F2mat> p);
void write_poly(F2vec P);
void write_poly_bin(F2vec &P);
#endif

int main()
{
    unsigned int size;
    cin >> size;
    vector<unsigned int> P_ints{};
    unsigned int x;
    for (int i = 0; i < size / 16; ++i)
    {
        cin >> hex >> x;
        P_ints.push_back(x);
    }
    F2vec P = from_ints(P_ints);
    //factor P in irreducible factors
    vector<F2vec> P_factors = factor_poly(P);

    F2vec A;
    F2vec B;

    vector<unsigned int> vecA;
    vector<unsigned int> vecB;

    stringstream sstr;
    vector<string> sols{};
    //iterate through all possible factorizations of P
    for (int i = 0; i != (1 << P_factors.size()); i++)
    {
        A = F2vec{1};
        B = F2vec{1};
        for (int j = 0; j < P_factors.size(); j++)
        {
            if ((i >> j) & 1)
                A = A * P_factors[j];
            else
                B = B * P_factors[j];
        }
        trim(A);
        trim(B);
        if ((A.size() <= size) && (B.size() <= size))
        {
            vecA = to_ints(A);
            vecB = to_ints(B);
            for (int i = 0; i < vecA.size(); i++)
            {
                if (i > 0)
                    sstr << " ";
                sstr << setfill('0') << setw(8) << hex << vecA[i];
            }
            for (int i = 0; i < vecB.size(); i++)
            {
                sstr << " " << setfill('0') << setw(8) << hex << vecB[i];
            }
            sols.push_back(sstr.str());
            sstr.str("");
            sstr.clear();
        }
    }
    sort(sols.begin(), sols.end());
    for (int i = 0; i < sols.size(); i++)
        cout << sols[i] << endl;
    return 0;
}

void trim(F2vec &P)
{
    if (!P.size())
        return;
    int n = P.size() - 1;
    while (!P[n])
    {
        n--;
        if (n < 0)
            break;
    };
    P.resize(n + 1);
}
void untrim(F2vec &P, int d)
{
    if (d < P.size())
        __throw_invalid_argument("error in untrim(F2vec&,int), dimension is smaller than input polynomial");
    while (P.size() < d)
        P.push_back(0);
}
F2vec operator^(F2vec P, F2vec Q)
{
    if (!P.size())
        return Q;
    else if (!Q.size())
        return P;

    int lP = P.size();
    int lQ = Q.size();
    int d = max(lP, lQ);
    F2vec R{};
    for (int i = 0; i < d; ++i)
        R.push_back(((i < lP) ? P[i] : 0) ^ ((i < lQ) ? Q[i] : 0));
    trim(R);
    return R;
}

F2vec operator<<(F2vec P, int s)
{
    if (!s)
        return P;
    F2vec R(s, 0);
    R.insert(R.end(), P.begin(), P.end());
    trim(R);
    return R;
}
F2vec operator>>(F2vec P, int s)
{
    if (!s)
        return P;
    else if (s > P.size())
        return F2vec{};
    F2vec::iterator it = P.begin();
    advance(it, s);
    F2vec R{};
    R.insert(R.begin(), it, P.end());
    trim(R);
    return R;
}
F2vec operator*(F2vec P, F2vec Q)
{
    if ((!Q.size()) || (!P.size()))
        return F2vec{};
    F2vec R{};
    if (P.size() < Q.size())
    {
        for (int i = 0; i < P.size(); i++)
            if (P[i])
                R = R ^ (Q << i);
    }
    else
    {
        for (int i = 0; i < Q.size(); i++)
            if (Q[i])
                R = R ^ (P << i);
    }
    trim(R);
    return R;
}

pair<F2vec, F2vec> divmod(F2vec P, F2vec Q)
{
    //special cases handling
    if (!Q.size())
    {
        __throw_domain_error("Error in divmod(F2vec,F2vec): attempted division by 0");
    }
    if (!P.size())
        return pair<F2vec, F2vec>(F2vec{}, F2vec{}); //0=0*Q+0
    else if (Q.size() == 1)
        return pair<F2vec, F2vec>{P, F2vec{}}; //P=1*P+0
    else if (Q.size() > P.size())
        return pair<F2vec, F2vec>{F2vec{}, P};

    F2vec R = P;
    F2vec D(P.size() - Q.size() + 1, 0);
    int shift = P.size() - Q.size();
    while ((shift >= 0) && (R.size() > 1))
    {
        R = R ^ (Q << shift);
        D[shift] = 1; //optional computation if only modulus is required
        shift = R.size() - Q.size();
    }
    return pair<F2vec, F2vec>(D, R);
}

F2vec operator%(F2vec P, F2vec Q)
{
    if (!Q.size())
    {
        __throw_domain_error("Error in operator%(F2vec,F2vec): attempted modulus by 0");
    }

    else if (Q.size() < 2)
        return F2vec{};
    trim(Q);
    F2vec R = P;

    int shift = P.size() - Q.size();
    while ((shift >= 0) && (R.size() > 1))
    {
        R = R ^ (Q << shift);
        shift = R.size() - Q.size();
    }

    trim(R);
    return R;
}

F2vec gcd(F2vec P, F2vec Q)
{
    trim(P);
    trim(Q);
    if (!Q.size())
        return P;
    else if ((Q.size() < 2) || (P.size() < 2))
        return F2vec{1};
    else
        return gcd(Q, P % Q);
}

vector<F2vec> factor_poly(F2vec P)
{
#ifdef __DEBUG__
    cout << "computing factorization for P=";
    write_poly(P);
    cout << endl;
#endif

    vector<F2vec> factors{};
    if (!P.size())
        __throw_domain_error("Error in factor_poly(F2vec): 0 is not factorizable");
    if (P.size() == 1)
        return vector<F2vec>{};
    if (P.size() == 2)
        return vector<F2vec>{P}; //X and X+1 are irreducible
    if (factorization_memo[P].size())
        return factorization_memo[P];
    //check P is square-free
    F2vec Pprime{};
    for (int i = 0; i + 1 < P.size(); i++)
        Pprime.push_back(P[i + 1] & ((i % 2) ^ 1));
    trim(Pprime);

#ifdef __DEBUG__
    cout << "P'= ";
    write_poly(Pprime);
    cout << endl;
#endif

    if (!Pprime.size())
    { //P is a square
        F2vec Q{};
        for (int i = 0; 2 * i < P.size(); i++)
            Q.push_back(P[2 * i]);
        trim(Q);

#ifdef __DEBUG__
        cout << "P is a square. sqrt(P)= ";
        write_poly(Q);
        cout << endl;
#endif

        vector<F2vec> Qfactors = factor_poly(Q); //factor sqrt(P)
        for (F2vec factor : Qfactors)
        {
            factors.push_back(factor); //double count each factor of the square root
            factors.push_back(factor);
        }
        factorization_memo[P] = factors;
        return factors;
    }
    else
    {
        F2vec G = gcd(P, Pprime);

#ifdef __DEBUG__
        cout << "P is not a square. gcd(P,P')= ";
        write_poly(G);
        cout << endl;
#endif

        if (G.size() > 1)
        { //P has a non-trivial square factor, G², so factor G and P/G separately
#ifdef __DEBUG__
            cout << "P is not square free." << endl;
            cout << "Now factoring G = ";
            write_poly(G);
            cout << endl;
#endif
            vector<F2vec> G_factors = factor_poly(G);

#ifdef __DEBUG__
            cout << "Now factoring P/G = ";
            write_poly(P / G);
#endif

            factors = factor_poly(P / G);
            for (F2vec factor : G_factors)
                factors.push_back(factor);
            factorization_memo[P] = factors;
            return factors;
        }
        else
        { //P is now assumed to be squared-free, so we can apply Berlekamp's algorithm
#ifdef __DEBUG__
            cout << "P is square free." << endl;
#endif

            int d = P.size() - 1;
            F2mat Phi_T{};
            F2vec row_vec{};
            F2vec b{1};
            F2vec phi_b;
            for (int row = 0; row < d; row++)
            {
                phi_b = (b << (2 * row)) % P; //compute X^(2*row) mod P(this is the frobenius map applied to the basis vector b in F2/(P))
                untrim(phi_b, d);
                Phi_T.push_back(phi_b);
            }
            F2mat M;
//Now we have the transpose matrix for the frobenius map
//We need to xor by the identity matrix
//and row-reduce it to read the fixed points mod P of the frobenius map
#ifdef __DEBUG__
            cout << "Phi_T dimensions / d: " << Phi_T.size() << " " << Phi_T[0].size() << " / " << d << endl;
            print_sq_mat(Phi_T);
            cout << endl;
#endif
            for (int i = 0; i < d; ++i)
                Phi_T[i][i] = !Phi_T[i][i]; //make (Phi-Id)_t
            //could use a call to sq_kernel, but this would require transposing Phi_T only to have it transposed again in the call
            pair<F2mat, F2mat> p = sq_row_reduced(Phi_T);

            vector<F2vec> fixed_points{};
#ifdef __DEBUG__
            cout << "Row reduction pair of Phi_T:" << endl;
            print_sq_reduction_pair(Phi_T);
            cout << endl;
#endif

            F2mat rref_A_T = p.first; //row reduced echeloned form A transpose if you please
            F2mat K = p.second;
            int row = d - 1;

            bool row_in_kernel = true;
            while (row_in_kernel)
            {
                for (int col = 0; col < d; col++)
                    row_in_kernel &= !rref_A_T[row][col];
                if (row_in_kernel)
                    fixed_points.push_back(K[row]);
                row--;
                if (row < 0)
                    break;
            }
#ifdef __DEBUG__
            cout << "fixed points of Phi mod P:" << endl;
            for (int i = 0; i < fixed_points.size(); i++)
            {
                write_poly(fixed_points[i]);
                cout << endl;
            }
#endif

            if (fixed_points.size() < 2)
            { //the only fixed points are F2, so P is irreducible
#ifdef __DEBUG__
                cout << "P is irreducible." << endl;
#endif
                factors.push_back(P);
                factorization_memo[P] = factors;
                return factors;
            }
            else
            {
                F2vec Q;
                int i = 0;
                bool found_non_trivial = false;
                while (!found_non_trivial)
                {
                    for (int col = 1; col < d; col++)
                    {
                        if (fixed_points[i][col])
                        {
                            Q = fixed_points[i];
                            found_non_trivial = true;
                            break;
                        }
                    }
                    i++;
                    if (i == fixed_points.size())
                        break;
                }
                trim(Q);

#ifdef __DEBUG__
                cout << "Found non-trivial fixed point:" << endl;
                write_poly(Q);
                cout << endl;
#endif

                //use P=gcd(P,Q)*gcd(P,Q+1)
                factors = factor_poly(gcd(P, Q));
                Q[0] = !Q[0];
                vector<F2vec> factors_alt = factor_poly(gcd(P, Q));
                for (F2vec factor : factors_alt)
                    factors.push_back(factor);
                factorization_memo[P] = factors;
                return factors;
            }
        }
    }
}
pair<F2mat, F2mat> sq_row_reduced(F2mat A) //O(n^3)
{
    int d = A.size();
    int pivot_col = 0;
    int pivot_row = 0;
    vector<bool> row_vec;
    F2mat I;
    for (int row = 0; row < d; row++)
    {
        row_vec.clear();
        for (int col = 0; col < d; col++)
        {
            row_vec.push_back(col == row);
        }
        I.push_back(row_vec);
    }

    while ((pivot_col < d) && (pivot_row < d))
    {
        int next_pivot_row = pivot_row;
        while (!A[next_pivot_row][pivot_col])
        {
            next_pivot_row++;
            if (next_pivot_row == d)
                break;
        }
        if (next_pivot_row == d)
            pivot_col++;
        else
        {
            swap(A[pivot_row], A[next_pivot_row]);
            swap(I[pivot_row], I[next_pivot_row]);

            for (int row = pivot_row + 1; row < d; row++)
            {
                if (A[row][pivot_col])
                {
                    A[row][pivot_col] = 0;
                    for (int col = pivot_col + 1; col < d; col++)
                        A[row][col] = A[row][col] ^ A[pivot_row][col];
                    for (int col = 0; col < d; col++)
                        I[row][col] = I[row][col] ^ I[pivot_row][col];
                    //For I we have to xor the full row, whereas we know A has zeros coefficients left of pivot_col
                }
            }
            pivot_col++;
            pivot_row++;
        }
    }

    return pair<F2mat, F2mat>(A, I);
}

vector<F2vec> sq_kernel(F2mat A)
{ //using ker(A)=orthogonal(span(A_transpose))
    int d = A.size();
    if (!d)
        __throw_domain_error("Error in sq_kernel(F2mat): computing kernel of empty matrix");
    else if (d < 2)
        return (A[0][0]) ? vector<F2vec>{} : vector<F2vec>{F2vec{1}};
    F2mat A_T;
    F2vec row_vec{};

    for (int row = 0; row < d; row++)
    { //build the transpose, O(n^2), so affordable compared to row reduction
        //(i.e. no need to rewrite a transpose version of the row reduction algorithm)
        row_vec.clear();
        for (int col = 0; col < d; col++)
            row_vec.push_back(A[col][row]);
        A_T.push_back(row_vec);
    }

    pair<F2mat, F2mat> p = sq_row_reduced(A_T);

#ifdef __DEBUG__
    cout << "Row reduction pair of A_T:" << endl;
    print_sq_reduction_pair(A_T);
    cout << endl;
#endif

    F2mat rref_A_T = p.first; //row reduced echeloned form A transpose
    F2mat P = p.second;

    vector<F2vec> basis{};
    int row = d - 1;

    bool row_in_kernel = true;
    while (row_in_kernel)
    {
        for (int col = 0; col < d; col++)
            row_in_kernel &= !rref_A_T[row][col];
        if (row_in_kernel)
            basis.push_back(P[row]);
        row--;
        if (row < 0)
            break;
    }
    return basis;
}

F2vec from_ints(vector<unsigned int> v)
{
    F2vec P{};
    for (int i = 0; i < v.size(); ++i)
    {
        unsigned int x = v[i];
        for (int b = 0; b < 32; ++b)
        {
            P.push_back(x & 1);
            x >>= 1;
        }
    }
    trim(P);
    return P;
}

vector<unsigned int> to_ints(F2vec P)
{
    vector<unsigned int> result{};
    trim(P);
    reverse(P.begin(), P.end());
    unsigned int x = 0;
    for (int i = 0; i < P.size(); ++i)
    {
        x = x | P[i];
        if (((i + 1) % 32 == 0) || (i + 1 == P.size()))
        {
            result.push_back(x);
            x = 0;
        }
        x <<= 1;
    }
    reverse(result.begin(), result.end());
    return result;
}
#ifdef __DEBUG__
void print_sq_mat_pair(pair<F2mat, F2mat> p)
{
    F2mat A = p.first;
    F2mat T = p.second;
    int N = A.size();

    for (int r = 0; r < N; r++)
    {
        for (int c = 0; c < N; c++)
        {
            if (c > 0)
                cout << " ";
            cout << A[r][c];
        }
        cout << " |";
        for (int c = 0; c < N; c++)
        {
            cout << " " << T[r][c];
        }
        cout << endl;
    }
}
void print_sq_reduction_pair(F2mat M)
{
    print_sq_mat_pair(sq_row_reduced(M));
}
void print_sq_mat(F2mat M)
{
    int N = M.size();

    for (int r = 0; r < N; r++)
    {
        for (int c = 0; c < N; c++)
        {
            if (c > 0)
                cout << " ";
            cout << M[r][c];
        }
        cout << endl;
    }
}

void write_poly(F2vec P)
{
    bool first = true;
    if (!P.size())
        cout << 0;
    else
    {
        for (int i = 0; i < P.size(); i++)
        {
            if (P[i])
            {
                if (!first)
                    cout << " + ";
                if (i == 0)
                    cout << 1;
                else
                    cout << "X^" << i;
                first = false;
            }
        }
    }
}

void write_poly_bin(F2vec &P)
{
    for (int i = P.size() - 1; i >= 0; i--)
        cout << P[i];
}
#endif

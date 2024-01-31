/*************************************************************************
    > File Name: train_framework.cpp
    > Description: distributed training framework based on ADMM
    > Author: Guozheng Wang
    > Mail: gzh.wang@outlook.com
    > Created Time: 2024-01-22
 ************************************************************************/

#include <mpi.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <stdarg.h>
#include "train_framework.h"
const double EPSILON = 1e-16;
using namespace std;

void
error(const char * const format, ...)
{
    va_list ap;
    va_start(ap,format);
    /* print out remainder of message */
    (void) vfprintf(stderr, format, ap);
    va_end(ap);
    (void) fprintf(stderr, "\n");
    (void) exit(EXIT_FAILURE);
}

void
coredump(const char * const format, ...)
{
    va_list ap;
    va_start(ap,format);
    /* print out remainder of message */
    (void) vfprintf(stderr, format, ap);
    va_end(ap);
    (void) fprintf(stderr, "\n");
    (void) abort();
}

void
warning(const char * const format, ...)
{
    va_list ap;
    va_start(ap,format);
    /* print out remainder of message */
    (void) vfprintf(stderr, format, ap);
    va_end(ap);
    (void) fprintf(stderr, "\n");
}

void
ensure(const bool condition,const char * const errorIfFail, ...)
{
    if (!condition) {
        va_list ap;
        va_start(ap,errorIfFail);
        /* print out remainder of message */
        (void) vfprintf(stderr, errorIfFail, ap);
        va_end(ap);
        (void) fprintf(stderr, "\n");
        (void) exit(EXIT_FAILURE);
    }
}
namespace comlkit{
    inline double min(double a, double b)
    {
        if (a < b) {
            return a;
        }
        else{
            return b;
        }
    }

    inline double max(double a, double b)
    {
        if (a > b) {
            return a;
        }
        else{
            return b;
        }
    }

    inline int sign(double x){
        return (0 < x) - (x < 0);
    }
    // Matrix.cc
    Matrix::Matrix() : m(0), n(0){
    }

    Matrix::Matrix(int m, int n) : m(m), n(n){
        matrix.reserve(n);
        for (int i = 0; i < m; i++) {
            Vector v(n, 0);
            matrix.push_back(v);
        }
    }

    Matrix::Matrix(int m, int n, int val) : m(m), n(n){
        matrix.reserve(n);
        for (int i = 0; i < m; i++) {
            Vector v(n, val);
            matrix.push_back(v);
        }
    }

    Matrix::Matrix(int m, int n, bool) : m(m), n(n){        // Identity Matrix constructor
        matrix.reserve(n);
        assert(m == n);         // works only for square matrices
        for (int i = 0; i < m; i++) {
            Vector v(n, 0);
            v[i] = 1;
            matrix.push_back(v);
        }
    }

    Matrix::Matrix(const Matrix& M) : m(M.m), n(M.n){
        matrix.reserve(n);
        Vector v(n, 0);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                v[j] = M(i, j);
            }
            matrix.push_back(v);
        }
    }

    double& Matrix::operator()(const int i, const int j){         // Access to element
        return matrix[i][j];
    }

    const double& Matrix::operator()(const int i, const int j) const {        // Const Access to element
        return matrix[i][j];
    }

    Vector& Matrix::operator[](const int i){         // Row access
        return matrix[i];
    }

    const Vector& Matrix::operator[](const int i) const {        // Row access
        return matrix[i];
    }

    Vector Matrix::operator()(const int i) const {        // Column Access (this is one is value only and const)
        Vector v(n, 0);
        for (int j = 0; j < n; j++)
        {
            v[j] = matrix[j][i];
        }
        return v;
    }

    void Matrix::push_back(const Vector& v){         // Add a row
        if (m == 0) {
            matrix.push_back(v);
            m++;
            n = v.size();
        }
        else{
            assert(v.size() == n);
            matrix.push_back(v);
            m++;
        }
    }

    void Matrix::remove(int i){
        assert((i >= 0) && (i < m));
        matrix.erase(matrix.begin()+i);
        m--;
    }

    int Matrix::numRows() const {
        return m;
    }

    int Matrix::numColumns() const {
        return n;
    }

    int Matrix::size() const {
        return m*n;
    }

    // VectorOperations.cc

    double sum(const Vector& x)
    {
        double s = 0.0;
        for (int i = 0; i < x.size(); i++)
        {
            s += x[i];
        }
        return s;
    }

// z = x + y.
    void vectorAddition(const Vector& x, const Vector& y, Vector& z)
    {
        // assert(n == y.size());
        z = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++)
        {
            z[i] = x[i] + y[i];
        }
        return;
    }

    void vectorFeatureAddition(const Vector& x, const SparseFeature& f, Vector& z)
    {
        // assert(x.size() == f.numFeatures);
        z = Vector(x);
        for (int i = 0; i < f.featureIndex.size(); i++)
        {
            int j = f.featureIndex[i];
            z[j] += f.featureVec[i];
        }
        return;
    }

    void vectorFeatureAddition(const Vector& x, const DenseFeature& f, Vector& z)
    {
        // assert(x.size() == f.featureVec.size());
        z = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++)
        {
            z[i] = x[i] + f.featureVec[i];
        }
        return;
    }

    void vectorScalarAddition(const Vector& x, const double a, Vector& z)
    {
        z = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++)
        {
            z[i] = x[i] + a;
        }
        return;
    }

// z = x - y
    void vectorSubtraction(const Vector& x, const Vector& y, Vector& z)
    {
        // assert(x.size() == y.size());
        z = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++)
        {
            z[i] = x[i] - y[i];
        }
        return;
    }

    void vectorFeatureSubtraction(const Vector& x, const SparseFeature& f, Vector& z)
    {
        // assert(x.size() == f.numFeatures);
        z = Vector(x);
        for (int i = 0; i < f.featureIndex.size(); i++)
        {
            int j = f.featureIndex[i];
            z[j] -= f.featureVec[i];
        }
        return;
    }

    void vectorFeatureSubtraction(const Vector& x, const DenseFeature& f, Vector& z)
    {
        // assert(x.size() == f.featureVec.size());
        z = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++)
        {
            z[i] = x[i] - f.featureVec[i];
        }
        return;
    }

    void vectorScalarSubtraction(const Vector& x, const double a, Vector& z)
    {
        z = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++)
        {
            z[i] = x[i] - a;
        }
        return;
    }

// z = x.*y, x and y vectors
    void elementMultiplication(const Vector& x, const Vector& y, Vector& z)
    {
        // assert(x.size() == y.size());
        z = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++)
        {
            z[i] = x[i]*y[i];
        }
        return;
    }

    Vector elementMultiplication(const Vector& x, const Vector& y)
    {
        Vector z;
        elementMultiplication(x, y, z);
        return z;
    }
// z = x.^a, x and y vectors
    void elementPower(const Vector& x, const double a, Vector& z)
    {
        z = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++)
        {
            z[i] = pow(x[i], a);
        }
        return;
    }

    Vector elementPower(const Vector& x, const double a)
    {
        Vector z;
        elementPower(x, a, z);
        return z;
    }

// z = a*x (a scalar)
    void scalarMultiplication(const Vector& x, const double a, Vector& z)
    {
        z = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++)
        {
            z[i] = a*x[i];
        }
        return;
    }

    void scalarMultiplication(const SparseFeature& f, const double a, SparseFeature& g)
    {
        g = f;
        for (int i = 0; i < f.featureIndex.size(); i++)
        {
            g.featureVec[i] = a*f.featureVec[i];
        }
        return;
    }

    void scalarMultiplication(const DenseFeature& f, const double a, DenseFeature& g)
    {
        g = f;
        for (int i = 0; i < f.featureVec.size(); i++)
        {
            g.featureVec[i] = a*f.featureVec[i];
        }
        return;
    }

    double innerProduct(const Vector& x, const Vector& y)
    {
        // assert(x.size() == y.size());
        double d = 0;
        for (int i = 0; i < x.size(); i++)
        {
            d += x[i]*y[i];
        }
        return d;
    }

    double featureProduct(const Vector& x, const SparseFeature& f)
    {
        // assert(x.size() == f.numFeatures);
        double d = 0;
        for (int i = 0; i < f.featureIndex.size(); i++)
        {
            int j = f.featureIndex[i];
            d += x[j]*f.featureVec[i];
        }
        return d;
    }

    double featureProduct(const Vector& x, const DenseFeature& f)
    {
        // assert(x.size() == f.featureVec.size());
        double d = 0;
        for (int i = 0; i < f.featureVec.size(); i++)
        {
            d += x[i]*f.featureVec[i];
        }
        return d;
    }
// An implementation of a feature-vector product, in the case when the feature dimension exceeds that of x.
    double featureProductCheck(const Vector& x, const SparseFeature& f)
    {
        // assert(x.size() == f.numFeatures);
        double d = 0;
        for (int i = 0; i < f.featureIndex.size(); i++)
        {
            int j = f.featureIndex[i];
            if (j < x.size())
                d += x[j]*f.featureVec[i];
        }
        return d;
    }

// An implementation of a feature-vector product, in the case when the feature dimension exceeds that of x.
    double featureProductCheck(const Vector& x, const DenseFeature& f)
    {
        // assert(x.size() == f.featureVec.size());
        double d = 0;
        for (int i = 0; i < x.size(); i++)
        {
            d += x[i]*f.featureVec[i];
        }
        return d;
    }

    void outerProduct(const Vector& x, const Vector& y, Matrix& m)
    {
        for (int i = 0; i < x.size(); i++)
        {
            Vector v(y.size(), 0);
            for (int j = 0; j < y.size(); j++) {
                v[j] = x[i]*y[j];
            }
            m.push_back(v);
        }
        return;
    }
// z = x - alpha*g
    void multiplyAccumulate(Vector& z, const Vector& x, const double alpha, const Vector& g)
    {
        // assert(x.size() == g.size());
        z = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++) {
            z[i] = x[i] - alpha*g[i];
        }
    }

// x = x - alpha*g
    void multiplyAccumulate(Vector& x, const double alpha, const Vector& g){
        // assert(x.size() == g.size());
        for (int i = 0; i < x.size(); i++) {
            x[i] -= alpha*g[i];
        }
    }

    int argMax(Vector& x) {
        double maxVal = 0.0;
        int maxIndex = -1;

        for (int i = 0; i < x.size(); i++) {
            if (maxVal < x[i]) {
                maxVal = x[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    double norm(const Vector& x, const int type)
    {
        double val = 0;
        for (int i = 0; i < x.size(); i++)
        {
            if (type == 0)         // l_0 norm
                val+= (x[i]==0);
            else if (type == 1)         // l_1 norm
                val+= std::abs(x[i]);
            else if (type == 2)         // l_2 norm
                val+= pow(x[i], 2);
            else if (type == 3)         // l_{\infty} norm
            {
                if (val < x[i])
                    val = x[i];
            }
        }
        if (type == 2)
            return sqrt(val);
        else
            return val;
    }

    Vector abs(const Vector& x){
        Vector absx(x.size(), 0);
        for (int i = 0; i < x.size(); i++) {
            absx[i] = std::abs(x[i]);
        }
        return absx;
    }

    Vector sign(const Vector& x){
        Vector sx(x.size(), 0);
        for (int i = 0; i < x.size(); i++) {
            sx[i] = sign(x[i]);
        }
        return sx;
    }

    void abs(const Vector& x, Vector& absx){
        absx = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++) {
            absx[i] = std::abs(x[i]);
        }
        return;
    }

    void sign(const Vector& x, Vector& sx){
        sx = Vector(x.size(), 0);
        for (int i = 0; i < x.size(); i++) {
            sx[i] = sign(x[i]);
        }
        return;
    }

//    void print(Vector& x){
//        for (int i = 0; i < x.size(); i++) {
//            std::cout<<x[i]<<" ";
//        }
//        std::cout<<"\n";
//    }
    Matrix matrixAddition(const Matrix& A, const Matrix& B)
    {
        assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
        Matrix C(A.numRows(), A.numColumns());
        for (int i = 0; i < A.numRows(); i++)
        {
            for (int j = 0; j < A.numColumns(); j++)
            {
                C(i, j) = A(i, j) + B(i, j);
            }
        }
        return C;
    }

    Matrix matrixSubtraction(const Matrix& A, const Matrix& B)
    {
        assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
        Matrix C(A.numRows(), A.numColumns());
        for (int i = 0; i < A.numRows(); i++)
        {
            for (int j = 0; j < A.numColumns(); j++)
            {
                C(i, j) = A(i, j) - B(i, j);
            }
        }
        return C;
    }

// z = A*x
    Vector leftMatrixVectorProduct(const Matrix& A, const Vector& x)
    {
        assert(A.numColumns() == x.size());
        Vector z(x.size(), 0);
        for (int i = 0; i < x.size(); i++)
        {
            z[i] = x*A[i];
        }
        return z;
    }

// z = x*A
    Vector rightMatrixVectorProduct(const Matrix& A, const Vector& x)
    {
        assert(A.numRows() == x.size());
        Vector z(A.numColumns(), 0);
        for (int i = 0; i < A.numColumns(); i++)
        {
            for (int j = 0; j < A.numRows(); j++) {
                z[i]+=A(j, i)*x[j];
            }
        }
        return z;
    }

// C = A*B
    Matrix matrixMatrixProduct(const Matrix& A, const Matrix& B)
    {
        assert(A.numColumns() == B.numRows());
        int dsize = A.numColumns();
        Matrix C(A.numRows(), B.numColumns());
        for (int i = 0; i < A.numRows(); i++)
        {
            for (int j = 0; j < B.numColumns(); j++) {
                for (int k = 0; k < dsize; k++) {
                    C(i, j) += A(i, k)*B(k, j);
                }
            }
        }
        return C;
    }

    const Matrix operator+(const Matrix& A, const Matrix &B){
        Matrix C = matrixAddition(A, B);
        return C;
    }

    const Matrix operator-(const Matrix& A, const Matrix &B){
        Matrix C = matrixSubtraction(A, B);
        return C;
    }

    const Vector operator*(const Matrix& A, const Vector &x){
        Vector z = leftMatrixVectorProduct(A, x);
        return z;
    }

    const Vector operator*(const Vector &x, const Matrix& A){
        Vector z = rightMatrixVectorProduct(A, x);
        return z;
    }
    const Matrix operator*(const Matrix& A, const Matrix& B){
        Matrix C = matrixMatrixProduct(A, B);
        return C;
    }
    const Matrix operator*(const Matrix& A, const double a){
        Matrix C(A.numRows(), A.numColumns());
        for (int i = 0; i < A.numRows(); i++)
        {
            for (int j = 0; j < A.numColumns(); j++)
            {
                C(i, j) = A(i, j) + a;
            }
        }
        return C;
    }

// x == y
    bool operator== (const Matrix& A, const Matrix& B){
        if ( (A.numRows() != B.numRows()) && (A.numColumns() == B.numColumns()) )
            return false;
        for (int i = 0; i < A.numRows(); i++) {
            if (A[i] != B[i])
                return false;
        }
        return true;
    }

// x != y
    bool operator!= (const Matrix& A, const Matrix& B){
        return !(A == B);
    }

    bool operator< (const Matrix& A, const Matrix& B){
        assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
        for (int i = 0; i < A.numRows(); i++)
        {
            if (A[i] < B[i])
                continue;
            else
                return false;
        }
        return true;
    }

    bool operator<= (const Matrix& A, const Matrix& B){
        assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
        for (int i = 0; i < A.numRows(); i++)
        {
            if (A[i] <= B[i])
                continue;
            else
                return false;
        }
        return true;
    }

    bool operator> (const Matrix& A, const Matrix& B){
        assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
        for (int i = 0; i < A.numRows(); i++)
        {
            if (A[i] > B[i])
                continue;
            else
                return false;
        }
        return true;
    }

    bool operator>= (const Matrix& A, const Matrix& B){
        assert((A.numRows() == B.numRows()) && (A.numColumns() == B.numColumns()));
        for (int i = 0; i < A.numRows(); i++)
        {
            if (A[i] >= B[i])
                continue;
            else
                return false;
        }
        return true;
    }
    template <size_t N>
    Vector assign(double (&array)[N]){
        Vector v(array, array+N);
        return v;
    }

    const Vector operator+(const Vector& x, const Vector &y){
        Vector z;
        vectorAddition(x, y, z);
        return z;
    }

    const Vector operator+(const Vector& x, const SparseFeature &f){
        Vector z;
        vectorFeatureAddition(x, f, z);
        return z;
    }

    const Vector operator+(const Vector& x, const DenseFeature &f){
        Vector z;
        vectorFeatureAddition(x, f, z);
        return z;
    }

    const Vector operator+(const Vector& x, const double a){
        Vector z;
        vectorScalarAddition(x, a, z);
        return z;
    }

    const Vector operator-(const Vector& x, const Vector &y){
        Vector z;
        vectorSubtraction(x, y, z);
        return z;
    }

    const Vector operator-(const Vector& x, const SparseFeature &f){
        Vector z;
        vectorFeatureSubtraction(x, f, z);
        return z;
    }

    const Vector operator-(const Vector& x, const DenseFeature &f){
        Vector z;
        vectorFeatureSubtraction(x, f, z);
        return z;
    }

    const Vector operator-(const Vector& x, const double a){
        Vector z;
        vectorScalarSubtraction(x, a, z);
        return z;
    }

    const double operator*(const Vector& x, const Vector &y){
        double d = innerProduct(x, y);
        return d;
    }

    const double operator*(const Vector& x, const SparseFeature &f){
        double d = featureProduct(x, f);
        return d;
    }

    const double operator*(const Vector& x, const DenseFeature &f){
        double d = featureProduct(x, f);
        return d;
    }

    const Vector operator*(const Vector& x, const double a){
        Vector z;
        scalarMultiplication(x, a, z);
        return z;
    }

    const Vector operator*(const double a, const Vector& x){
        Vector z;
        scalarMultiplication(x, a, z);
        return z;
    }

    const SparseFeature operator*(const SparseFeature& f, const double a){
        SparseFeature g;
        scalarMultiplication(f, a, g);
        return g;
    }

    const SparseFeature operator*(const double a, const SparseFeature& f){
        SparseFeature g;
        scalarMultiplication(f, a, g);
        return g;
    }

    const DenseFeature operator*(const DenseFeature& f, const double a){
        DenseFeature g;
        scalarMultiplication(f, a, g);
        return g;
    }

    const DenseFeature operator*(const double a, const DenseFeature& f){
        DenseFeature g;
        scalarMultiplication(f, a, g);
        return g;
    }

    Vector& operator+=(Vector& x, const Vector &y){
        // assert(x.size() == y.size());
        for (int i = 0; i < x.size(); i++)
        {
            x[i] += y[i];
        }
        return x;
    }

    Vector& operator+=(Vector& x, const SparseFeature &f){
        // assert(x.size() == f.numFeatures);
        for (int i = 0; i < f.featureIndex.size(); i++)
        {
            int j = f.featureIndex[i];
            x[j] += f.featureVec[i];
        }
        return x;
    }

    Vector& operator+=(Vector& x, const DenseFeature &f){
        // assert(x.size() == f.featureVec.size());
        for (int i = 0; i < x.size(); i++)
        {
            x[i] += f.featureVec[i];
        }
        return x;
    }

    Vector& operator+=(Vector& x, const double a){
        for (int i = 0; i < x.size(); i++)
        {
            x[i] += a;
        }
        return x;
    }

    Vector& operator-=(Vector& x, const Vector &y){
        // assert(x.size() == y.size());
        for (int i = 0; i < x.size(); i++)
        {
            x[i] -= y[i];
        }
        return x;
    }

    Vector& operator-=(Vector& x, const SparseFeature &f){
        // assert(x.size() == f.numFeatures);
        for (int i = 0; i < f.featureIndex.size(); i++)
        {
            int j = f.featureIndex[i];
            x[j] -= f.featureVec[i];
        }
        return x;
    }

    Vector& operator-=(Vector& x, const DenseFeature &f){
        // assert(x.size() == f.featureVec.size());
        for (int i = 0; i < x.size(); i++)
        {
            x[i] -= f.featureVec[i];
        }
        return x;
    }

    Vector& operator-=(Vector& x, const double a){
        for (int i = 0; i < x.size(); i++)
        {
            x[i] -= a;
        }
        return x;
    }

    Vector& operator*=(Vector& x, const double a){
        for (int i = 0; i < x.size(); i++)
        {
            x[i] *= a;
        }
        return x;
    }

// x == y
    bool operator== (const Vector& x, const Vector& y){
        if (x.size() != y.size())
            return false;
        for (int i = 0; i < x.size(); i++) {
            if (x[i] != y[i])
                return false;
        }
        return true;
    }

// x != y
    bool operator!= (const Vector& x, const Vector& y) {
        return !(x == y);
    }

    bool operator< (const Vector& x, const Vector& y) {
        // assert(x.size() == y.size());
        for (int i = 0; i < x.size(); i++)
        {
            if (x[i] >= y[i])
                return false;
        }
        return true;
    }

    bool operator<= (const Vector& x, const Vector& y) {
        // assert(x.size() == y.size());
        for (int i = 0; i < x.size(); i++)
        {
            if (x[i] > y[i])
                return false;
        }
        return true;
    }

    bool operator> (const Vector& x, const Vector& y) {
        // assert(x.size() == y.size());
        for (int i = 0; i < x.size(); i++)
        {
            if (x[i] <= y[i])
                return false;
        }
        return true;
    }

    bool operator>= (const Vector& x, const Vector& y) {
        // assert(x.size() == y.size());
        for (int i = 0; i < x.size(); i++)
        {
            if (x[i] < y[i])
                return false;
        }
        return true;
    }

    std::ostream& operator<<(std::ostream& os, const Vector& x)
    {
        for (int i = 0; i < x.size(); i++) {
            os << x[i] << " ";
        }
        return os;
    }
//    long GetFileSize(const char * filename )
//    {
//        struct stat statebuf;
//
//        if ( stat( filename, &statebuf ) == -1 )
//            return -1L;
//        else
//            return statebuf.st_size;
//    }
//
//    Vector readVector(char* File, int n){
//        Vector v;
//        double tmpd;
//        ifstream iFile;
//        printf("Reading list of labels from %s...\n", File);
//        iFile.open(File, ios::in);
//        if (!iFile.is_open()) {
//            printf("Error: Cannot open file\n");
//        }
//        else {
//            for (int i=0; i<n; i++) {
//                iFile >> tmpd;
//                v.push_back(tmpd);
//            }
//        }
//        iFile.close();
//        return v;
//    }
//
//    Vector readVectorBinary(char* File, int n){
//        Vector v;
//        int unitsize = 8;
//        double tmpd;
//        FILE* fp;
//        if(!(fp=fopen(File,"rb"))) {
//            printf("ERROR: cannot open file %s",File);
//        }
//
//        for(int i=0; i<n; i++) {
//            fread(&tmpd,unitsize,1,fp);
//            v.push_back(tmpd);
//        }
//        fclose(fp);
//        return v;
//    }
//
//// A specific implementation to read a binary matrix stored as floats.
//    vector<vector<float> > readKernelfromFileFloat(char* graphFile, int n){
//        vector<vector<float> > kernel;
//        kernel.resize(n);
//        for(int i=0; i<n; i++) {kernel[i].resize(n);}
//        int unitsize=4;
//        string tmp;
//        ifstream iFile;
//        int count = 0; // row count
//        FILE* fp;
//        float tmpf;
//        printf("Loading graph from %s...\n",graphFile);
//        if (!(fp=fopen(graphFile,"rb"))) {
//            error("ERROR: cannot open file %s\n",graphFile);
//        }
//        int nRow = long(GetFileSize(graphFile)/n)/unitsize;
//        printf("Number of rows: %d\n", nRow);
//        for (int i = 0; i < nRow; i++) {
//            for (int j = 0; j < n; j++) {
//                fread(&tmpf,unitsize,1,fp);
//                kernel[count+i][j] = tmpf;
//            }
//        }
//        count += nRow;
//        fclose(fp);
//        printf("Finished loading the graph from %s...\n",graphFile);
//        return kernel;
//    }
//
//// A specific implementation to read a binary matrix stored as doubles.
//    vector<vector<float> > readKernelfromFileDouble(char* graphFile, int n){
//        vector<vector<float> > kernel;
//        kernel.resize(n);
//        for(int i=0; i<n; i++) {kernel[i].resize(n);}
//        int unitsize=8;
//        string tmp;
//        ifstream iFile;
//        int count = 0; // row count
//        FILE* fp;
//        double tmpd;
//        printf("Loading graph from %s...\n",graphFile);
//        if (!(fp=fopen(graphFile,"rb"))) {
//            printf("ERROR: cannot open file %s",graphFile);
//        }
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < n; j++) {
//                fread(&tmpd,unitsize,1,fp);
//                kernel[i][j] = (float)tmpd;
//            }
//        }
//        fclose(fp);
//        return kernel;
//    }
//
//
//// Helper Function to read in the Feature based functions.
//    int line2words(char *s, struct SparseFeature & Feature, int & max_feat_idx){
//        int digitwrd;
//        float featureval;
//        int pos = 0;
//        int numUniqueWords = 0;
//        max_feat_idx = 0;
//        while (sscanf(s,"%d %f %n",&digitwrd, &featureval, &pos) == 2) {
//            s += pos;
//            numUniqueWords++;
//            if (digitwrd < 0) {
//                cout << "The input feature graph is not right: " << " some feature index is <0, please make sure the feature indices being above 0\n";
//            }
//            assert((digitwrd >= 0));
//            if (digitwrd > max_feat_idx) {
//                max_feat_idx = digitwrd;
//            }
//            /*if (featureval < 0){
//               cout << "The input feature graph has a feature value being negative, please make sure that the feature graph has all feature value >= 0\n";
//               }*/
//            //assert(featureval >= 0);
//            Feature.featureIndex.push_back(digitwrd);
//            Feature.featureVec.push_back(featureval);
//        }
//        return numUniqueWords;
//    }

//// A specific implementation to read in a feature file, with number of features in first line.
//    std::vector<struct SparseFeature> readFeatureVectorSparse(char* featureFile, int& n, int &numFeatures){
//        // read the feature based function
//        std::vector<struct SparseFeature> feats;
//        // feats.resize(n);
//        ifstream iFile;
//        FILE *fp = NULL;
//        char line[300000]; //stores information in each line of input
//        printf("Reading feature File from %s...\n", featureFile);
//        if ((fp = fopen(featureFile, "rt")) == NULL) {
//            printf("Error: Cannot open file %s", featureFile);
//            exit(-1);
//        }
//        long int lineno = 0; // row number
//        // read in the first line of the file, and skip it.
//        //fgets(line,sizeof(line),fp);
//        int pos = 0;
//        n = 0;
//        numFeatures = 0;
//        //sscanf(line,"%d %d %n",&n,&numFeatures, &pos);
//        //cout<<"n = "<<n<<" and "<< "numFeatures = "<<numFeatures<<"\n";
//        int max_feat_idx;
//        while ( fgets(line,sizeof(line),fp) != NULL) {
//            feats.push_back(SparseFeature());
//            feats[lineno].index = lineno;
//            feats[lineno].numUniqueFeatures = line2words(line, feats[lineno], max_feat_idx); //line2words transforms input with fmt digwords:featurevals into initialization of structure "Feature"
//            //feats[lineno].numFeatures = numFeatures;
//            if (max_feat_idx > numFeatures) {
//                numFeatures = max_feat_idx;
//            }
//            lineno++;
//        }
//        fclose(fp);
//        n = lineno; // row number
//        numFeatures++; // sample dimension
//        printf("The input feature file has %d instances and the dimension of the features is %d\n", n, numFeatures);
//        for (int idx = 0; idx < n; idx++) {
//            feats[idx].numFeatures = numFeatures;
//        }
//
//        cout<<"done with reading the feature based file\n";
//        return feats;
//    }

// Read labels and features stored in LIBSVM format.
    void readFeatureLabelsLibSVM( const char* fname, std::vector<struct SparseFeature>& features, Vector& y, int& n, int &numFeatures)
    {
        features = std::vector<struct SparseFeature>();
        FILE* file;
        printf("Reading feature File from %s\n", fname);
        if ((file = fopen(fname, "rt")) == NULL) {
            printf("Error: Cannot open file %s", fname);
            exit(-1);
        }
        float label; bool init = true;
        char tmp[ 1024 ];
        numFeatures = 0;
        n = 0;
        struct SparseFeature feature;
        while( fscanf( file, "%s", tmp ) == 1 ) {
            int index; float value;
            if( sscanf( tmp, "%d:%f", &index, &value ) == 2 ) {
                feature.featureIndex.push_back(index-1); feature.featureVec.push_back(value); // 修改了索引对应的模型位置
                if (index > numFeatures)
                    numFeatures = index;
            }else{
                if( !init ) {
                    y.push_back(label);
                    features.push_back(feature);
                    feature = SparseFeature();
                    n++;
                }
                assert(sscanf( tmp, "%f", &label ) == 1);
                init = false;
            }
        }
        y.push_back(label);
        features.push_back(feature);
        n++;
        numFeatures++;
        printf("The input feature file has %d instances and the dimension of the features is %d\n", n, numFeatures);
        for (int idx = 0; idx < features.size(); idx++) {
            features[idx].numFeatures = numFeatures;
        }
        fclose(file);
    }

//// 2016-12-28
//// A specific implementation to read in a feature file, with number of features in first line.
//// Split dataset into a test set and training set
//    void readFeatureVectorSparseCrossValidate(char* featureFile, char* labelFile,
//                                              int& numTrainingInstances, int &numFeatures,
//                                              float percentTrain,
//                                              std::vector<struct SparseFeature> &trainFeats,
//                                              std::vector<struct SparseFeature> &testFeats,
//                                              Vector &trainLabels,
//                                              Vector &testLabels)
//    {
//        // read the feature based function
//        std::vector<struct SparseFeature> feats;
//        // feats.resize(n);
//        ifstream iFile;
//        FILE *fp = NULL;
//        char line[300000]; //stores information in each line of input
//        printf("Reading feature File from %s...\n", featureFile);
//        if ((fp = fopen(featureFile, "rt")) == NULL) {
//            printf("Error: Cannot open file %s", featureFile);
//            exit(-1);
//        }
//        long int lineno = 0;
//        // read in the first line of the file, and skip it.
//        //fgets(line,sizeof(line),fp);
//        int pos = 0;
//        int n = 0;
//        numFeatures = 0;
//        //sscanf(line,"%d %d %n",&n,&numFeatures, &pos);
//        //cout<<"n = "<<n<<" and "<< "numFeatures = "<<numFeatures<<"\n";
//        int max_feat_idx;
//        while ( fgets(line,sizeof(line),fp) != NULL) {
//            feats.push_back(SparseFeature());
//            feats[lineno].index = lineno;
//            feats[lineno].numUniqueFeatures = line2words(line, feats[lineno], max_feat_idx); //line2words transforms input with fmt digwords:featurevals into initialization of structure "Feature"
//            //feats[lineno].numFeatures = numFeatures;
//            if (max_feat_idx > numFeatures) {
//                numFeatures = max_feat_idx;
//            }
//            lineno++;
//        }
//        fclose(fp);
//        n = lineno;
//        numFeatures++;
//        printf("The input feature file has %d instances and the dimension of the features is %d\n", n, numFeatures);
//        for (int idx = 0; idx < n; idx++) {
//            feats[idx].numFeatures = numFeatures;
//        }
//
//        cout<<"Done with reading the feature based file.\n";
//
//        // Split into testing and training data
//        // First gather labels
//        Vector labels = readVector(labelFile, n);
//
//        // if the number of instances were listed before the instances, we could do this in one pass
//        numTrainingInstances = percentTrain * (float) n;
//
//        cout <<"Splitting into " << numTrainingInstances << " training instances and " <<
//             n - numTrainingInstances << " testing instances.\n";
//
//        std::vector<int> instances;
//        for(int i = 0; i < n; i++) {
//            instances.push_back(i);
//        }
//        srand(time(NULL));
//        std::random_shuffle (instances.begin(), instances.end());
//
//        // set aside first numTrainingInstances instances as training data, remaining n-numTrainingInstances instances
//        // as test set
//        int curr_feature_idx = 0;
//        for (std::vector<int>::iterator it = instances.begin(); it != instances.end(); it++) {
//            if( curr_feature_idx < numTrainingInstances) {
//                trainFeats.push_back(feats[*it]);
//                trainLabels.push_back(labels[*it]);
//            } else {
//                testFeats.push_back(feats[*it]);
//                testLabels.push_back(labels[*it]);
//            }
//            curr_feature_idx++;
//        }
//    }
// continuous_functions
    ContinuousFunctions::ContinuousFunctions(bool isSmooth) : isSmooth(isSmooth){
        m = 0; n = 0;
    }
    ContinuousFunctions::ContinuousFunctions(bool isSmooth, int m, int n) : isSmooth(isSmooth), m(m), n(n){
    }
    ContinuousFunctions::ContinuousFunctions(const ContinuousFunctions& c) : isSmooth(c.isSmooth), m(c.m), n(c.n) {
    }

    ContinuousFunctions::~ContinuousFunctions(){
    }

    double ContinuousFunctions::eval(const Vector& x) const {
        return 0;
    }

    Vector ContinuousFunctions::evalGradient(const Vector& x) const {  // in case the function is non-differentiable, this is the subgradient
        Vector gradient(m, 0);
        Vector xdiff(x);
        for (int i = 0; i < m; i++) {
            xdiff[i]+=EPSILON;
            gradient[i] = (eval(xdiff) - eval(x))/EPSILON;
            xdiff[i]-=EPSILON;
        }
        return gradient;
    }

    void ContinuousFunctions::eval(const Vector& x, double& f, Vector& gradient) const {
        gradient = evalGradient(x);
        f = eval(x);
        return;
    }

    Vector ContinuousFunctions::evalStochasticGradient(const Vector& x, std::vector<int>& batch) const {
        return evalGradient(x);
    }

    void ContinuousFunctions::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
        return eval(x, f, g);
    }
    Matrix ContinuousFunctions::evalHessian(const Vector& x) const {
        Matrix hessian;
        Vector xdiff(x);
        for (int i = 0; i < m; i++) {
            xdiff[i]+=EPSILON;
            hessian.push_back(evalGradient(xdiff) - evalGradient(x));
            xdiff[i]-=EPSILON;
        }
        return hessian;
    }

    void ContinuousFunctions::evalHessianVectorProduct(const Vector& x, const Vector& v, Vector& Hxv) const {
        Matrix hessian = evalHessian(x);
        Hxv = hessian*v;
    }

    double ContinuousFunctions::operator()(const Vector& x) const
    {
        return eval(x);
    }

    int ContinuousFunctions::size() const {  // number of features or dimension size
        return m;
    }

    int ContinuousFunctions::length() const {  // number of convex functions adding up
        return n;
    }
// L2ProbitLoss.cc
    template <class Feature>
    L2ProbitLoss<Feature>::L2ProbitLoss(int m, std::vector<Feature>& features, Vector& y, Vector& sum_msg, int num_neighbor, double lambda, double rho) :
            ContinuousFunctions(true, m, features.size()), features(features), y(y), sum_msg_(sum_msg), num_neighbor_(num_neighbor), lambda(lambda), rho_(rho)
    {
        if (n > 0)
            assert(features[0].numFeatures == m);
        assert(features.size() == y.size());
    }

    template <class Feature>
    L2ProbitLoss<Feature>::L2ProbitLoss(const L2ProbitLoss& l) :
            ContinuousFunctions(true, l.m, l.n), features(l.features), y(l.y), lambda(l.lambda), sum_msg_(l.sum_msg_) {
    }

    template <class Feature>
    L2ProbitLoss<Feature>::~L2ProbitLoss(){
    }

    template <class Feature>
    double L2ProbitLoss<Feature>::eval(const Vector& x) const {
        assert(x.size() == m);
        double sum = 0.5*lambda*(x*x);
        for (int i = 0; i < n; i++) {
            double val = y[i]*(x*features[i])/sqrt(2);
            double probitval = (1/2)*(1 + erf(val))+EPSILON;
            sum-= log(probitval);
        }
        return sum;
    }

    template <class Feature>
    Vector L2ProbitLoss<Feature>::evalGradient(const Vector& x) const {
        assert(x.size() == m);
        Vector g = lambda*x;
        for (int i = 0; i < n; i++) {
            double val = y[i]*(x*features[i])/sqrt(2);
            double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
            double probitval = (1/2)*(1 + erf(val))+EPSILON;
            g -= features[i]*(y[i]*normval/probitval);
        }
        return g;
    }

    template <class Feature>
    void L2ProbitLoss<Feature>::eval(const Vector& x, double& f, Vector& g) const {
//	assert(x.size() == m);
//	g = lambda*x;
//	f = 0.5*lambda*(x*x);
        f = 0;
        g = 2 * rho_ * num_neighbor_ * x + sum_msg_;
        for (int i = 0; i < n; i++) {
            double val = y[i]*(x*features[i])/sqrt(2);
            double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
            double probitval = 0.5*(1 + erf(val))+EPSILON;
            g -= features[i]*(y[i]*normval/probitval);
            f -= log(probitval);
        }
        f += rho_ * num_neighbor_ * x * x + x * sum_msg_;
        return;
    }

    template <class Feature>
    Vector L2ProbitLoss<Feature>::evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const {
        assert(x.size() == m);
        Vector g = lambda*x;
        double val;
        for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
            double val = y[*it]*(x*features[*it])/sqrt(2);
            double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
            double probitval = (1/2)*(1 + erf(val))+EPSILON;
            g -= features[*it]*(y[*it]*normval/probitval);
        }
        return g;
    }

    template <class Feature>
    void L2ProbitLoss<Feature>::evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const {
        assert(x.size() == m);
        g = lambda*x;
        f = 0.5*lambda*(x*x);
        for (vector<int>::iterator it = miniBatch.begin(); it != miniBatch.end(); it++) {
            double val = y[*it]*(x*features[*it])/sqrt(2);
            double normval = (1/sqrt(2*M_PI))*exp(-(val*val));
            double probitval = (1/2)*(1 + erf(val))+EPSILON;
            g -= features[*it]*(y[*it]*normval/probitval);
            f -= log(probitval);
        }
        return;
    }
    template class L2ProbitLoss<SparseFeature>;
    template class L2ProbitLoss<DenseFeature>;
    Vector gdNesterov(const ContinuousFunctions& c, const Vector& x0, double alpha, const double gamma,
                      const int maxEval, const double TOL, bool resetAlpha, bool useinputAlpha, int verbosity){
        Vector x(x0);
        Vector g;
        double f;
        c.eval(x, f, g); // 更新f和g
        Vector y = x;
        Vector xnew;
        double fnew;
        Vector gnew;
        double t, tnew;
        double gnorm = norm(g);
        int funcEval = 1;
        if (!useinputAlpha)
            alpha = 1/norm(g);
        while ((gnorm >= TOL) && (funcEval < maxEval) )
        {
            if (funcEval > 1) {
                tnew = (1 + sqrt(1 + 4*t*t))/2;
                y = xnew + ((t - 1)/tnew)*(xnew - x);
                x = xnew;
                t = tnew;
                c.eval(y, f, g);
                funcEval++;
            }

            xnew = y - alpha*g; // 更新x
            c.eval(xnew, fnew, gnew);
            funcEval++;

            double gg = g*g;
            // double fgoal = f - gamma*alpha*gg;
            // Backtracking line search
            while (fnew > f - gamma*alpha*gg) {
                alpha = alpha*alpha*gg/(2*(fnew + gg*alpha - f));
                xnew = y - alpha*g;
                c.eval(xnew, fnew, gnew);
                funcEval++;
            }
            if (resetAlpha)
                alpha = min(1, 2*(f - fnew)/gg);

            gnorm = norm(gnew);
//        if(myid == 0){
//            if (verbosity > 0)
//                printf("numIter: %d, alpha: %e, ObjVal: %e, OptCond: %e\n", funcEval, alpha, fnew, gnorm);
//        }
        }
        return xnew;
    }
}

using namespace comlkit;

conf_util::conf_util() {
    string conf_file = "../group.conf";
    parse(conf_file);
}

// Parse configuration file.
void conf_util::parse(const string &conf_file) {
    ifstream confIn(conf_file.c_str());
    string line;
    vector<string> vitems;
    while (getline(confIn, line)) {
        vitems.clear();
        if (line.empty() || line[0] == '#')
            continue;
        const int len = line.length();
        char s[len + 1];
        strcpy(s, line.c_str());
        char *pos = strtok(s, " =");
        int32_t k = 0;
        while (pos != NULL) {
            vitems.push_back(pos);
            pos = strtok(NULL, "=");
            k++;
        }
        if (k != 2) {
            cout << "args conf error!" << endl;
            exit(0);
        }
        conf_items.insert({vitems[0], vitems[1]});
    }
}

/// Return parameter
/// \tparam T
/// \param item_name
/// \return
template<class T>
T conf_util::getItem(const std::string &item_name) {
    stringstream sitem;
    T result;
    sitem << conf_items[item_name];
    sitem >> result;
    return result;
}

args_t::args_t(int rank, int size) {
    myid = rank;
    procnum = size;
    rho = 0.0;
    data_direction_ = "./data";
}

void args_t::get_args() {
    conf_util admm_conf;
    rho = admm_conf.getItem<double>("rho");
}

void args_t::print_args() {
    cout << "#************************configuration***********************";
    cout << endl;
    cout << "#Number of processors: " << procnum << endl;
    cout << "#Train data: " << train_data_path << endl;
    cout << "#Test data: " << test_data_path << endl;
    cout << "#Max iteration: " << maxIteration << endl;
    cout << "#Update_method: " << Update_method << endl;
    cout << "#Comm_method: " << Comm_method << endl;
    cout << "#Node of per Group: " << nodesOfGroup << endl;
    cout << "#rho: " << rho << endl;
    cout << "#************************************************************";
    cout << endl;
}

std::string &LeftTrim(std::string &s) {
    auto it = s.begin();
    for (; it != s.end() && std::isspace(*it); ++it);
    s.erase(s.begin(), it);
    return s;
}

std::string &RightTrim(std::string &s) {
    auto it = s.end() - 1;
    for (; it != s.begin() - 1 && std::isspace(*it); --it);
    s.erase(it + 1, s.end());
    return s;
}

std::string &Trim(std::string &s) {
    return RightTrim(LeftTrim(s));
}
// Properties.cpp
Properties::Properties(const std::string &path)
{
    ParseFromFile(path);
}
void Properties::ParseFromFile(const std::string &path) {
    std::ifstream reader(path);
    if (reader.fail()) {
        std::cout << "无法打开配置文件，文件名：" << path<<std::endl;
    }

    // 新建一个map临时存放属性值
    std::map<std::string, std::string> temp;
    std::string line;
    while (std::getline(reader, line)) {
        // 每一行中#号后面的内容为注释，因此删去这些内容
        std::size_t pos = line.find_first_of('#');
        if (pos != std::string::npos) {
            line.erase(pos);
        }
        // 除去每一行内容的前后空格
        Trim(line);
        if (line.empty()) {
            continue;
        }
        // 每一行内容的格式为key:value，冒号两边可以有空格
        pos = line.find_first_of(':');
        if (pos == std::string::npos || pos == 0 || pos == line.length() - 1) {
            std::cout << "格式错误，应该为key:value格式，" << line<<std::endl;
        }
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        Trim(key);
        Trim(value);
        temp[key] = value;
    }
    reader.close();

    //命令行参数的优先程度大于配置文件的参数，因此只把properties中没有的参数复制过去
    for (auto it = temp.begin(); it != temp.end(); ++it) {
        if (properties_.count(it->first) == 0) {
            properties_[it->first] = it->second;
        }
    }
}

std::string Properties::GetString(const std::string &property_name) {
    return properties_.at(property_name);
}

int Properties::GetInt(const std::string &property_name) {
    return Convert<int, std::string>(properties_.at(property_name));
}

double Properties::GetDouble(const std::string &property_name) {
    return Convert<double, std::string>(properties_.at(property_name));
}

//bool Properties::GetBool(const std::string &property_name) {
//    if (properties_.at(property_name) == "true") {
//        return true;
//    } else if (properties_.at(property_name) == "false") {
//        return false;
//    }
//    //LOG(FATAL) << property_name << " must be true or false." << std::endl;
//    return false;
//}
//
//bool Properties::HasProperty(const std::string &property_name) {
//    return properties_.count(property_name) != 0;
//}
//
//void Properties::CheckProperty(const std::string &property_name) {
//    if (!HasProperty(property_name)) {
//        //LOG(FATAL) << "Missing parameter " << property_name << std::endl;
//    }
//}
//
//void Properties::Print() {
//    //LOG(INFO) << "**************************************";
//    for (auto it = properties_.begin(); it != properties_.end(); ++it) {
//        //LOG(INFO) << it->first << ":" << it->second;
//    }
//    //LOG(INFO) << "**************************************";
//}



//int sayhello(MPI_Comm comm) {
//    int size, rank;
//    int value;
//    MPI_Status status;
//    char pname[MPI_MAX_PROCESSOR_NAME]; int len;
//    if (comm == MPI_COMM_NULL) {
//        printf("You passed MPI_COMM_NULL !!!\n");
//        return -1;
//    }
//    MPI_Comm_size(comm, &size);
//    MPI_Comm_rank(comm, &rank);
//    MPI_Get_processor_name(pname, &len);
//    pname[len] = 0;
//    printf("Hello, World! I am process %d of %d on %s.\n",
//           rank, size, pname);
//    int x;
//    if(rank == 0){
//        value = 2;
//        MPI_Send(&value, 1, MPI_INT, 1, 0, comm);
//    }
//    else{
//        MPI_Recv(&value, 1, MPI_INT, 0, 0, comm, &status);
//    }
//    return value;
//}
// GroupStrategy correct!!!!!!!!!!
GroupStrategy::GroupStrategy(int nums) {
    repeatIter = 3;
    this->repeatIter = nums;
}

/// \param data  Node sorting in a single iteration.
/// \param GroupNum  Group count.
/// \param Group1   Group 1 of switching nodes.
/// \param Group2   Group 2 of switching nodes.
/// \param part     Which part 0, the first half and the second half are exchanged?
vector<int> GroupStrategy::exchangeElement(vector<int> data, int GroupNum, int Group1, int Group2, int part) {
    int nums = data.size() / GroupNum;// Number of worker in each group.
    // The packet vector to exchange.
    vector<int> vec1, vec2;
    for (int i = nums * Group1; i < nums * (Group1 + 1); i++)
        vec1.push_back(data[i]);
    for (int i = nums * Group2; i < nums * (Group2 + 1); i++)
        vec2.push_back(data[i]);
    int index = 0;
    // Grouped parts: 0 is the first half and 1 is the second half.
    if (part == 1)
        index = nums / 2;
    for (int i = nums * Group1 + (nums / 2) * part; i < nums * (Group1 + 1) - (nums / 2) * (1 - part); i++) {
        data[i] = vec2[index++]; // Switch worker in groups, and switch before and after according to part.
    }
    if (part == 1)
        index = nums / 2;
    else
        index = 0;
    for (int i = nums * Group2 + (nums / 2) * part; i < nums * (Group2 + 1) - (nums / 2) * (1 - part); i++) {
        data[i] = vec1[index++];
    }
    return data;
}

///
/// \param nodes Node list.
/// \param groupNums Grouping number.
/// \return
vector<vector<int>> GroupStrategy::divideGroup(vector<int> nodes, int groupNums) {
    vector<vector<int>> returnAns;
    int nodesNums = nodes.size(); // Total number of nodes.
//    int numsOfGroup=nodesNums/groupNums; // Generally, it is set to the total number of process nodes opened for single or multiple machine nodes.
    vector<int> tempVec;// Temporary worker vector.
    int iter = 0;
    for (int i = 0; i < nodes.size(); i++) {
        tempVec.push_back(nodes[i]);
    }
    returnAns.push_back(tempVec);
    tempVec.clear(); // Empty array.
    ++iter;
    int part = 0;
    int exchange = groupNums / 2;
    int u = 0;
    // Multiple packet exchanges to generate initialization packets.
    while (iter < repeatIter) {
        vector<int> temp;
        if ((exchange * u + 1) % groupNums == 0)
            temp = exchangeElement(returnAns[iter - 1], groupNums, 0, (exchange * u + 2) % groupNums, (part++) % 2);
        else
            temp = exchangeElement(returnAns[iter - 1], groupNums, 0, (exchange * u + 1) % groupNums, (part++) % 2);
        for (int i = 2; i < groupNums - 1; i++) {
            if ((exchange * u + i + 1) % groupNums == i)
                temp = exchangeElement(returnAns[iter - 1], groupNums, i, (exchange * u + 2) % groupNums, (part++) % 2);
            else
                temp = exchangeElement(temp, groupNums, i, (exchange * u + i + 1) % groupNums, (part++) % 2);
        }
        returnAns.push_back(temp);
        iter++;
        if (groupNums != 2)
            part++;
        u++;
    }
    return returnAns;
}

/// Judge the smallest index node in vec (excluding 0), and the smaller it is, the faster it will be.
/// \param vec
/// \param index
/// \return
int GroupStrategy::position(double *vec, int size, int index) {
    int ans = 0;
    for (int i = 0; i < size; i++) {
        if (vec[i] != 0) {
            if (vec[i] < vec[index])
                ans++;
        }
    }
    return ans;
}

/// Find the fast node.
/// \param time The subscript of time is the id of the fast node.
/// \param group
/// \param node The requesting node needs to be excluded.
/// \param node Need to find numsofGrup-1 fast nodes.
/// \return Need to return the subscript of the group in the current iteration.
vector<int> GroupStrategy::findFastNodes(double *time, vector<int> group, int node, int numsofGrup, int size) {
    vector<int> fastnodes;
    // Choose the sorting idea, sort the worker according to the computing speed of nodes, and find out the fast nodes.
    for (int i = 0; i < size - 1; i++) {
        int index = i;
        for (int j = i + 1; j < size; j++) {
            if (time[j] < time[index]) {
                index = j;
            }
        }
        // Exchange the time corresponding to I and index.
        int temp;
        temp = time[index];
        time[index] = time[i];
        time[i] = temp;
        if (index == node)
            continue;
        for (int j = 0; j < group.size(); j++) {
            if (index == group[j]) {
                fastnodes.push_back(j);
                break;
            }
        }
        if (fastnodes.size() == numsofGrup - 1)
            break;
    }
    return fastnodes;
}

/// Replace the nodes in the grouping with slow and fast nodes.
/// \param vec Grouping vector.
/// \param node
/// \param fastVec  Subscript of fast node.
/// \param numsOfgroup How many nodes are there in each group?
void GroupStrategy::changeGroup(vector<vector<int>> &data, int node, vector<int> fastVec, int numsOfgroup, int iter) {
    vector<int> vec;
    vec = data[(iter - 1) % repeatIter];
    for (int i = 0; i < data[(iter - 1) % repeatIter].size(); i++)
        vec.push_back(data[(iter - 1) % repeatIter][i]);
    int index = 0;
    for (index; index < vec.size(); index++) {
        if (vec[index] == node)
            break;
    }
    int j = 0;
    for (int i = index / numsOfgroup * numsOfgroup; i < (1 + index / numsOfgroup) * numsOfgroup; i++) {
        // Exchange
        if (i != index) {
            int temp = vec[i];
            vec[i] = vec[fastVec[j]];
            vec[fastVec[j++]] = temp;
        }
    }
    data.push_back(vec);
}

void GroupStrategy::MasterNodes(int procnum, int nodesOfGroup, int DynamicGroup, int maxIteration) {
    double *node_beforeTime;
    double *node_afterTime;
    double *node_caltime; // It is judged as a slow node for the packet recorder to record a single calculation time of each node.
    vector<int> nodes;
    vector<vector<int>> Group;
    // Accept the communication request generated by the group.
    int nodetemp;
    int iter = 0;
    MPI_Status status;
    int iterTemp = 0;
    int c = 1;
    int *sendNodes;
    // Record the calculation time, and judge the slow nodes by the calculation time.
    node_caltime = new double[procnum - 1];
    node_beforeTime = new double[procnum - 1];
    node_afterTime = new double[procnum - 1];
    sendNodes = new int[nodesOfGroup];
    for (int i = 0; i < procnum - 1; i++) {
        node_caltime[i] = 0.0;
        node_beforeTime[i] = 0.0;
        node_afterTime[i] = 0.0;
    }
    // Predefine grouping rules and initialize grouping.
    for (int i = 0; i < procnum - 1; i++)
        nodes.push_back(i);
    Group = divideGroup(nodes, (procnum - 1) / nodesOfGroup);
    while (true) {
        MPI_Probe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        nodetemp = status.MPI_SOURCE;
        MPI_Recv(&iter, 1, MPI_INT, nodetemp, 1, MPI_COMM_WORLD, &status);
        vector<int> tempVec;
        if (DynamicGroup == 1) {
            // Re-modify some grouping methods according to the node that sent for the first time in each iteration and the last calculation time.
            // Assign fast nodes to fast nodes according to the first node and assign fast nodes to slow nodes. Never mind the first iteration.
            // Never mind the first 3 iterations.
            if (iter > repeatIter && iter > iterTemp) {
                iterTemp = iter;
                if ((iter - 1) % repeatIter ==
                    0) { // Every iteration of repeatIter, return to the original grouping form.
                    Group.push_back(Group[0]);
                } else {
                    // Only fast nodes and slow nodes need to be processed.
                    // Modify half the nodes in each group.
                    int pos = position(node_caltime, procnum - 1,
                                       nodetemp); // Find the number of fast worker by calculating the time.
                    if (pos < (procnum - 1) / 4 || pos >= (procnum - 1) / 4 *
                                                          3) { // Don't worry about the middle one, it is divided into four parts, and the first group and the last group need to assign him fast nodes.
                        cout << "iter " << iter << " change Group !!!" << endl;
                        // Modify Group[iter-1] and modify other nodes in the group where nodetemp belongs to be fast nodes.
                        vector<int> fastnodex = findFastNodes(node_caltime, Group[(iter - 1) % repeatIter], nodetemp,
                                                              nodesOfGroup, procnum - 1);
                        changeGroup(Group, nodetemp, fastnodex, nodesOfGroup, iter);
                        //Group.push_back(newvectemp);
                    } else {
                        Group.push_back(Group[(iter - 1) % repeatIter]);
                    }
                }
            }
            // Update the iteration interval information of all nodes.
            node_beforeTime[nodetemp] = node_afterTime[nodetemp];
            node_afterTime[nodetemp] = (double) (clock()) / CLOCKS_PER_SEC;
            node_caltime[nodetemp] = node_afterTime[nodetemp] - node_beforeTime[nodetemp];
            tempVec = Group[iter - 1];
            //tempVec=Group[(iter-1)%3];
        } else {
            tempVec = Group[(iter - 1) % repeatIter];
        }
        int u = 0;
        for (u = 0; u < procnum; u++) {
            if (tempVec[u] == nodetemp) {
                break;
            }
        }
        int tempIndex = 0;
        //cout<<nodetemp<<":";
        for (int v = u / nodesOfGroup * nodesOfGroup; v < (u / nodesOfGroup + 1) * nodesOfGroup; v++) {
            //cout<<tempVec[v]<<" ";
            sendNodes[tempIndex++] = tempVec[v];
        }
        //cout<<endl;
        MPI_Send(sendNodes, nodesOfGroup, MPI_INT, nodetemp, 2, MPI_COMM_WORLD);
        c++;
        if (c > maxIteration * (procnum - 1)) {
            break;
        }
    }
    delete[] node_caltime;
    delete[] node_beforeTime;
    delete[] node_afterTime;
    delete[] sendNodes;
}

void neighbors::setNeighbours(int nums, int *set) {
    neighborsNums=nums;
    for(int i=0;i<nums;i++)
    {
        neighs[i]=set[i];
    }
}

void neighbors::clearNeighbours()
{
    this->neighborsNums==0;
}

ADMM::ADMM(args_t *args, vector<struct SparseFeature> train_features, comlkit::Vector ytrain,
           vector<struct SparseFeature> test_features, comlkit::Vector ytest, int dimension, int optimizer, double beta) { // jensen lib
    // MPI
    //    CreateGroup();
    myid = args->myid;
    procnum = args->procnum;
    // Dataset settings.
    train_features_ = train_features;
    ytrain_ = ytrain;
    mtrian_ = train_features_[0].numFeatures;
    ntrian_ = train_features_.size();
    if (myid == 0) {
        test_features_ = test_features;
        ytest_ = ytest;
        mtest_ = test_features_[0].numFeatures;
        ntest_ = test_features_.size();
    }
    dim_ = dimension;
    data_number_ = ntrian_;
    // ADMM parameter setting and initialization.
    rho = args->rho;
    beta_ = 0.2; // high precision && faster convergence
    optimizer_ = optimizer;
    comlkit::Vector new_x(dim_, 0);
    comlkit::Vector sum_msg(dim_, 0);
    comlkit::Vector new_x_old(dim_, 0);
    comlkit::Vector new_alpha(dim_, 0);
    comlkit::Vector new_alpha_old(dim_, 0);
    new_x_ = new_x;
    sum_msg_ = sum_msg;
    new_x_old_ = new_x_old;
    new_alpha_ = new_alpha;
    new_alpha_old_ = new_alpha;
    maxIteration = args->maxIteration;
    nodesOfGroup = args->nodesOfGroup;// nodesOfGroup = 8
    worker_ranks_ = new int[nodesOfGroup - 1];
    nears.neighborsNums = nodesOfGroup; // Number of nodes in the group, including this node.
    nears.neighs = new int[nears.neighborsNums];
}

void ADMM::CreateGroup() {
    int color_odd, color_even;
    color_odd = myid / nears.neighborsNums;
    MPI_Comm_split(MPI_COMM_WORLD, color_odd, myid, &SUBGRP_COMM_ODD_);
    int subgrp_rank_odd, subgrp_size_odd;
    MPI_Comm_rank(SUBGRP_COMM_ODD_, &subgrp_rank_odd);
    MPI_Comm_size(SUBGRP_COMM_ODD_, &subgrp_size_odd);
    color_even = myid % nears.neighborsNums;
    MPI_Comm_split(MPI_COMM_WORLD, color_even, myid, &SUBGRP_COMM_EVEN_);
    int subgrp_rank_even, subgrp_size_even;
    MPI_Comm_rank(SUBGRP_COMM_EVEN_, &subgrp_rank_even);
    MPI_Comm_size(SUBGRP_COMM_EVEN_, &subgrp_size_even);
}

ADMM::~ADMM() {
    delete[] worker_ranks_;
}

void ADMM::alpha_update(const comlkit::Vector &new_x, const comlkit::Vector &old, const comlkit::Vector &sumx) {
    int numsofneighbors = nears.neighborsNums - 1;
//    new_alpha_ += rho * (numsofneighbors * new_x - sumx) - beta_ * (new_alpha_ - old); // bad
    new_alpha_ += rho * (numsofneighbors * new_x - sumx) - beta_ * (new_x - old); // primal variable history message is better than dual variable
//    new_alpha_ += rho * (numsofneighbors * new_x - sumx) + beta_ * (new_x - old); // svr regression
//    new_alpha_ +=
//            rho * (numsofneighbors * new_x - sumx); // primal variable history message is better than dual variable
//    new_alpha_ += rho * (numsofneighbors * new_x - sumx);
}

double ADMM::predict_comlkit(int method) {
    double p = 0.1;
    double error = 0;
    if (ytest_.empty()) {
        return 0;
    } else {
        int counter = 0;
        int notuse = 0;
        int sample_num = ytest_.size();
        for (int i = 0; i < sample_num; ++i) {
            // logistic regression
            if (method == 1) {
                double val = 1.0 / (1 + exp(-1 * new_x_ * test_features_[i]));
                if (val >= 0.5 && ytest_[i] == 1) {
                    counter++;
                } else if (val < 0.5 && ytest_[i] == -1) {
                    counter++;
                }
            } else if (method == 2) {
                // svm
                double val = new_x_ * test_features_[i];
                if (val >= 0 && ytest_[i] == 1) {
                    counter++;
                } else if (val < 0 && ytest_[i] == -1) {
                    counter++;
                }
            } else if (method == 3) {
                // svr
                error += pow(((new_x_ * test_features_[i]) - ytest_[i]), 2);
            } else if (method == 4) {
                counter++;
            }
        }
        if (method == 3) {
            return pow(error / sample_num, 0.5); // RMSE of svr
        } else {
            return counter * 100.0 / sample_num; // svm & lr prediction accuracy.
        }
    }
}

double ADMM::loss_value_comlkit(int method) {
    double p = 0.1;
    if (ytest_.empty()) {
        return 0;
    } else {
        double sum = 0;
        int sample_num = test_features_.size();
        // LR loss
        if (method == 1) {
            for (int i = 0; i < sample_num; ++i) {
                sum += std::log(1 + std::exp(-ytest_[i] * new_x_ * test_features_[i])); // logistic regression loss
            }
        } else if (method == 2) {
            // SVM loss
            for (int i = 0; i < sample_num; i++) {
                double preval = ytest_[i] * (new_x_ * test_features_[i]);
                if (1 - preval >= 0) {
                    sum += (1 - preval) * (1 - preval);
                }
            }
        } else if (method == 3) {
            // SVR loss
            for (int i = 0; i < sample_num; ++i) {
                double preval = (new_x_ * test_features_[i]) - ytest_[i];
                if (preval < -p) {
                    sum += (preval + p) * (preval + p);
                } else if (preval > p) {
                    sum += (preval - p) * (preval - p);
                }
            }
        } else if (method == 4) {
            // Probit loss
            for (int i = 0; i < sample_num; i++) {
                double val = ytest_[i] * (new_x_ * test_features_[i]) / sqrt(2);
                double probitval = 0.5 * (1 + erf(val)) + EPSILON;
                sum -= log(probitval);
            }
        }
        return sum / sample_num;
    }
}

void ADMM::group_train(clock_t start_time) {
    // Instantiate the communication time and calculate the time variable.
    double b_time, e_time;
    double comm_btime, comm_etime, cal_btime, cal_etime;
    double comm_time, cal_time, fore_time = 0.0;
    MPI_Status status;
    int32_t k = 1;
    vector<int> nodes;
    double sparseCount = 0;
    if (myid == 0)
        printf("%3s %12s %12s %12s %12s %12s %12s\n", "#", "accuracy or RMES", "loss", "single_iter_time", "comm_time",
               "cal_time",
               "sum_time");
    b_time = start_time;
    // Create Torus group.
    //    CreateGroup();
    // Experimental results are saved to a csv file.
    FILE *fp = fopen("calculation_results.csv", "w+");
    if (fp == NULL) {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }
    while (k <= maxIteration) {
        // Request group generation from the group generator Send the current iteration where it is located.
        MPI_Send(&k, 1, MPI_INT, procnum - 1, 1, MPI_COMM_WORLD);
        // Get the generated group.
        MPI_Probe(procnum - 1, 2, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &nears.neighborsNums); // Get node's neighbor states.
        MPI_Recv(nears.neighs, nears.neighborsNums, MPI_INT, procnum - 1, 2, MPI_COMM_WORLD,
                 &status); // Receive a vector containing INT.
        // Grouping method, can be divided into groups.
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Group worker_group;
        MPI_Group_incl(world_group, nears.neighborsNums, nears.neighs, &worker_group);
        MPI_Comm worker_comm;
        MPI_Comm_create_group(MPI_COMM_WORLD, worker_group, 0, &worker_comm);
        comm_time = 0;
        // Torus分组
        if (k != 1) {
            for (int i = 0; i < dim_; i++) {
                sum_msg_[i] = new_alpha_[i] -
                              rho *
                              ((nears.neighborsNums - 1) * new_x_[i] +
                               sum_msg_[i]); // 本worker与邻居worker聚合的x_j{j\in neighs}
            }
        }
        new_x_old_ = new_x_;
        new_alpha_old_ = new_alpha_;
        // Instantiating the correspondence problem form.
//        LogisticLoss<SparseFeature> ll(mtrian_, train_features_, ytrain_, sum_msg_, nears.neighborsNums-1, 1, rho, myid);
//        L2LeastSquaresLoss<SparseFeature> le(mtrian_, features_, label_, sum_msg_, nears.neighborsNums, 1, rho);
//        L2LogisticLoss<SparseFeature> l2(mtrian_, train_features_, ytrain_, sum_msg_, nears.neighborsNums-1, 1, rho);
//        L2SmoothSVRLoss<SparseFeature> svr(mtrian_, train_features_, ytrain_, sum_msg_, nears.neighborsNums, 1, rho);
//        L2SmoothSVMLoss<SparseFeature> svm(mtrian_, train_features_, ytrain_, sum_msg_, nears.neighborsNums, 1, rho);
        L2ProbitLoss<SparseFeature> probit(mtrian_, train_features_, ytrain_, sum_msg_, nears.neighborsNums, 1, rho);
        // Subproblem Solving Optimizer
        cal_btime = MPI_Wtime();
//        if (k <= 2) {
//            new_x_ = gdNesterov(svr, new_x_, 1, 1e-4, 50);
//        } else {
//            new_x_ = gdNesterov(svr, new_x_, 1, 1e-4, 10);
//        }
//        new_x_ = tron(probit, myid, new_x_, 50);
//        new_x_ = cg(svr, new_x_, 1, 1e-4, 50);
        new_x_ = gdNesterov(probit, new_x_, 1, 1e-4, 100);
//        new_x_ = gd(probit, new_x_, 0.1, 100);
//        new_x_ = gdLineSearch(svr, new_x_, 1, 1e-4, 50);
//        new_x_ = lbfgsMin(probit, new_x_, 1, 1e-4, 50);
//        new_x_ = TRON(l2, train_features_, ytrain_, new_x_);
//        new_x_ = gdBarzilaiBorwein(svr, new_x_, 1, 1e-4, 50);
//        new_x_ = gdNesterov(svr, new_x_, 1, 1e-4, 50);
//        new_x_ = SVRDual(train_features_, ytrain_, 2, 1, 0.1, 1e-3, 100);
//        new_x_ = sgdAdagrad(svm, new_x_, ntrian_, 1e-2, 200, 1e-4, 10);
//        new_x_ = sgdDecayingLearningRate(svm, new_x_, ntrian_, 0.5 * 1e-1, 200, 1e-4, 100);
        cal_etime = MPI_Wtime();
        // Model parameter synchronization
        double *new_x_temp = new double[dim_];
        double *sum_msg_temp = new double[dim_];
        for (int i = 0; i < dim_; ++i) {
            new_x_temp[i] = new_x_[i];
            sum_msg_temp[i] = sum_msg_[i];
        }
        comm_btime = MPI_Wtime();
        MPI_Allreduce(new_x_temp, sum_msg_temp, dim_, MPI_DOUBLE, MPI_SUM, worker_comm);
        // Torus synchronizaiton method.
//        if (k % 2 == 0) {
//            MPI_Allreduce(new_x_temp, sum_msg_temp, dim_, MPI_DOUBLE, MPI_SUM, SUBGRP_COMM_EVEN_);
//        } else {
//            MPI_Allreduce(new_x_temp, sum_msg_temp, dim_, MPI_DOUBLE, MPI_SUM, SUBGRP_COMM_ODD_);
//        }
        comm_etime = MPI_Wtime();
        for (int i = 0; i < dim_; ++i) {
            sum_msg_temp[i] -= new_x_temp[i];
            new_x_[i] = new_x_temp[i];
            sum_msg_[i] = sum_msg_temp[i];
        }
        alpha_update(new_x_, new_x_old_, sum_msg_);
        delete[] new_x_temp;
        delete[] sum_msg_temp;
        e_time = MPI_Wtime();

        cal_time = (double) (cal_etime - cal_btime);
//        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) {
            double sum_time = (double) (e_time - b_time);
            cal_time = (double) (cal_etime - cal_btime);
            comm_time = (double) (comm_etime - comm_btime);
            sparseCount = 0;
            // Model sparsity calculation
//                for (int i = 0; i < dim; i++) {
//                    if (new_x_[i] < 1e-5) {
//                        sparseCount++;
//                    }
//                }
            // Output of calculation results to the console
            double predict = predict_comlkit(1);
            double loss = loss_value_comlkit(1);
            printf("%3d %12f %12f %12f %12f %12f %12f\n", k, predict, loss,
                   sum_time - fore_time,
                   comm_time, cal_time, sum_time);
            // Calculation results are stored to a file
            fprintf(fp, "%d %f %f %f %f %f %f\n", k, predict, loss,
                    sum_time - fore_time,
                    comm_time, cal_time, sum_time);
            fore_time = sum_time;
            sum_comm_ += comm_time;
        }
        k++;
    }
    fclose(fp);
}

void test_main(MPI_Comm comm) {
    int myid, procnum;
    double start_time, end_time;
    char filename[100];
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procnum);
//    // Two ways to get information from the configuration file and store it in the argument
    args_t *args = new args_t(myid, procnum); // Way 1
    Properties properties("../group.conf");
    double rho = properties.GetDouble("rho");
    int maxIteration = properties.GetInt("maxIteration");
    int nodesOfGroup = properties.GetInt("nodesOfGroup");
    int DynamicGroup = properties.GetInt("DynamicGroup");
    int QuantifyPart = properties.GetInt("QuantifyPart");
    int Comm_method = properties.GetInt("Comm_method");
    int Update_method = properties.GetInt("Update_method");
    int Repeat_iter = properties.GetInt("repeat_iter");
    int Sparse_comm = properties.GetInt("sparse_comm");
    int optimizer;
    double beta;
//    std::string train_data_path = properties.GetString("train_data_path");
//    std::string test_data_path = properties.GetString("test_data_path");
//
//    // The last node acts as a group generator. No file reads are required.
    if (myid == procnum - 1) {
        GroupStrategy group_trategy(Repeat_iter);
        group_trategy.MasterNodes(procnum, nodesOfGroup, DynamicGroup, maxIteration);
        MPI_Finalize();
    } else {
////         Read training set and test set files.
////        sprintf(filename, "/mirror/dataset/log1p/%d/data%03d", procnum - 1, myid);
////        char const *testdata_file = "/mirror/dataset/log1p/test";
////        sprintf(filename, "/mirror/dataset/log1p/%d/data%03d", procnum - 1, myid);
////        char const *testdata_file = "/mirror/dataset/log1p/test";
////        sprintf(filename, "/mirror/dataset/real/%d/data%03d", procnum - 1, myid);
////        char const *testdata_file = "/mirror/dataset/real/test";
////        sprintf(filename, "/mirror/dataset/gisette/%d/data%03d", procnum - 1, myid);
////        char const *testdata_file = "/mirror/dataset/gisette/test";
////        sprintf(filename, "/mirror/dataset/news20old/%d/data%03d", procnum - 1, myid);
////        char const *testdata_file = "/mirror/dataset/news20old/test";
        sprintf(filename, "/mirror/dataset/rcv1/%d/data%03d", procnum - 1, myid); // 注意是否多1个节点。
        char const *testdata_file = "/mirror/dataset/rcv1/test";
        vector<struct SparseFeature> train_features, test_features;
        comlkit::Vector ytrain, ytest;
        int ntrian, mtrian, ntest, mtest;
        readFeatureLabelsLibSVM(filename, train_features, ytrain, ntrian, mtrian);
        if (myid == 0) {
            readFeatureLabelsLibSVM(testdata_file, test_features, ytest, ntest, mtest);
        }
//        // Ensure that the Consistency of model parameter dimensions between processes.
        int temp = train_features[0].numFeatures;
        if (myid == 0) {
            temp = max(temp,
                       test_features[0].numFeatures); // select the max dim between train feature and test feature.
        }
//        vector<int> nodelist;
//        for (int i = 0; i < procnum - 1; i++) // 注意是否多1个节点。
//            nodelist.push_back(i);
//        spar::SimpleAllreduce<spar::MaxOperator, int>(&temp, 1, myid, nodelist, MPI_COMM_WORLD);
//        for (int i = 0; i < train_features.size(); ++i) {
//            train_features[i].numFeatures = temp;
//        }
        // Parameters read in the Properties class are assigned to the args_t class, redundant operation.
        args->maxIteration = maxIteration;
        args->nodesOfGroup = nodesOfGroup;
        args->rho = rho;
        args->Comm_method = Comm_method;
        args->Update_method = Update_method;
        args->Repeat_iter = Repeat_iter;
        // Instantiate admm and assign a value.
        ADMM admm(args, train_features, ytrain, test_features, ytest, temp, optimizer, beta);
        admm.sum_cal_ = 0.0;
        admm.sum_comm_ = 0.0;
        admm.quantify_part_ = QuantifyPart;
        admm.dynamic_group_ = DynamicGroup;
        admm.update_method_ = Update_method;
        admm.sparse_comm_ = Sparse_comm;
        // Exporting relevant configuration information.
//        args->print_args();
        if (myid == 0)
            args->print_args();
        start_time = MPI_Wtime();
        admm.group_train(start_time);
//        end_time = MPI_Wtime();
//        if (myid == 0) {
//            double temp = (double) (end_time - start_time);
//            cout << "run time: " << temp << "  "
//                 << "comm time:" << admm.sum_comm_ << "  "
//                 << "cal  time:" << temp - admm.sum_comm_ << endl;
//        }
        MPI_Finalize();
    }
}

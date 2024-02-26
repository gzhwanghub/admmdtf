#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <cstring>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <assert.h>
#include <stdlib.h>
using namespace std;
// error.h
//__BEGIN_DECLS
//
///* Print a message with `fprintf (stderr, FORMAT, ...)';
//   if ERRNUM is nonzero, follow it with ": " and strerror (ERRNUM).
//   If STATUS is nonzero, terminate the program with `exit (STATUS)'.  */
//
//extern void error (int __status, int __errnum, const char *__format, ...)
//__attribute__ ((__format__ (__printf__, 3, 4)));
//
//extern void error_at_line (int __status, int __errnum, const char *__fname,
//                           unsigned int __lineno, const char *__format, ...)
//__attribute__ ((__format__ (__printf__, 5, 6)));
//
///* If NULL, error will flush stdout, then print on stderr the program
//   name, a colon and a space.  Otherwise, error will call this
//   function without parameters instead.  */
//extern void (*error_print_progname) (void);
//
///* This variable is incremented each time `error' is called.  */
//extern unsigned int error_message_count;
//
///* Sometimes we want to have at most one error per line.  This
//   variable controls whether this mode is selected or not.  */
//extern int error_one_per_line;
//
//#include <bits/floatn.h>
//#if defined __LDBL_COMPAT || __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 1
//# include <bits/error-ldbl.h>
//#else
///* Do not inline error and error_at_line when long double has the same
//   size of double, nor when long double reuses the float128
//   implementation, because that would invalidate the redirections to the
//   compatibility functions.  */
//# if defined __extern_always_inline && defined __va_arg_pack
//#  include <bits/error.h>
//# endif
//#endif
//
//__END_DECLS

namespace comlkit{
    typedef std::vector<double> Vector;
    struct SparseFeature {        //Stores the feature vector for each item in the groundset
        long int index;      // index of the item
        int numUniqueFeatures;      // number of non-zero enteries in the feature vector
        std::vector<int> featureIndex;      // Indices which are non-zero (generally sparse)
        std::vector<double> featureVec;      // score of the features present.
        int numFeatures;
    };
    struct DenseFeature { //Stores the feature vector for each item in the groundset
        long int index; // index of the item
        std::vector<double> featureVec; // score of the dense feature vector.
        int numFeatures;
    };
    // Matrix.h
    class Matrix {
    protected:
        std::vector<std::vector<double> > matrix;
        int m;
        int n;
    public:
        Matrix();
        Matrix(int m, int n);
        Matrix(int m, int n, int val);
        Matrix(int m, int n, bool);
        Matrix(const Matrix& M);
        double& operator()(const int i, const int j);         // point access
        const double& operator()(const int i, const int j) const;         // const point access
        Vector& operator[](const int i);         // row access
        const Vector& operator[](const int i) const;         // const row access
        Vector operator()(const int i) const;         // const column access (read only)
        void push_back(const Vector& v);         // add a row at the end
        void remove(int i);         // delete a row at position i (i starts from 0)
        int numRows() const;
        int numColumns() const;
        int size() const;
    };
    // FileIO.h
//    long GetFileSize(const char *filename);
//
//    Vector readVector(char *File, int n);
//
//    Vector readVectorBinary(char *File, int n);
//
//// A specific implementation to read a binary matrix stored as floats.
//    std::vector<std::vector<float> > readKernelfromFileFloat(char *graphFile, int n);
//
//// A specific implementation to read a binary matrix stored as doubles.
//    std::vector<std::vector<float> > readKernelfromFileDouble(char *graphFile, int n);
//
//    int line2words(char *s, struct SparseFeature &Feature, int &maxID);
//
//    std::vector<struct SparseFeature> readFeatureVectorSparse(char *featureFile, int &n, int &numFeatures);

    void readFeatureLabelsLibSVM(const char *fname, std::vector<struct SparseFeature> &features, Vector &y, int &n,
                                 int &numFeatures);

//    void readFeatureVectorSparseCrossValidate(char *featureFile, char *labelFile,
//                                              int &numTrainingInstances, int &numFeatures,
//                                              float percentTrain,
//                                              std::vector<struct SparseFeature> &trainFeats,
//                                              std::vector<struct SparseFeature> &testFeats,
//                                              Vector &trainLabels,
//                                              Vector &testLabels);
// VectorOperations.h
    double sum(const Vector& x);
    void vectorAddition(const Vector& x, const Vector& y, Vector& z);
    void vectorFeatureAddition(const Vector& x, const SparseFeature& f, Vector& z);
    void vectorFeatureAddition(const Vector& x, const DenseFeature& f, Vector& z);
    void vectorScalarAddition(const Vector& x, const double a, Vector& z);
    void vectorSubtraction(const Vector& x, const Vector& y, Vector& z);
    void vectorFeatureSubtraction(const Vector& x, const SparseFeature& f, Vector& z);
    void vectorFeatureSubtraction(const Vector& x, const DenseFeature& f, Vector& z);
    void vectorScalarSubtraction(const Vector& x, const double a, Vector& z);
    void elementMultiplication(const Vector& x, const Vector& y, Vector& z);
    Vector elementMultiplication(const Vector& x, const Vector& y);
    Vector elementPower(const Vector& x, const double a);
    void elementPower(const Vector& x, const double a, Vector& z);
    void scalarMultiplication(const Vector& x, const double a, Vector& z);
    void scalarMultiplication(const SparseFeature& f, const double a, SparseFeature& g);
    void scalarMultiplication(const DenseFeature& f, const double a, DenseFeature& g);
    double innerProduct(const Vector& x, const Vector& y);
    double featureProduct(const Vector& x, const SparseFeature& f);
    double featureProduct(const Vector& x, const DenseFeature& f);
    double featureProductCheck(const Vector& x, const SparseFeature& f);
    double featureProductCheck(const Vector& x, const DenseFeature& f);
    void outerProduct(const Vector& x, const Vector& y, Matrix& m);
    int argMax(Vector& x);
    double norm(const Vector& x, const int type = 2);               // default is l_2 norm
//    void print(const Vector& x);
    Vector abs(const Vector& x);
    Vector sign(const Vector& x);
    void abs(const Vector& x, Vector& absx);
    void sign(const Vector& x, Vector& sx);
    void multiplyAccumulate(Vector& z, const Vector& x, const double alpha, const Vector& g);
    void multiplyAccumulate(Vector& x, const double alpha, const Vector& g);

    template <size_t N> Vector assign(double (&array)[N]);

    const Vector operator+(const Vector& x, const Vector &y);
    const Vector operator+(const Vector& x, const SparseFeature& f);
    const Vector operator+(const Vector& x, const DenseFeature& f);
    const Vector operator+(const Vector& x, const double a);
    const Vector operator-(const Vector& x, const Vector &y);
    const Vector operator-(const Vector& x, const SparseFeature& f);
    const Vector operator-(const Vector& x, const DenseFeature& f);
    const Vector operator-(const Vector& x, const double a);
    const double operator*(const Vector& x, const Vector &y);
    const double operator*(const Vector& x, const SparseFeature &f);
    const double operator*(const Vector& x, const DenseFeature &f);
    const Vector operator*(const Vector& x, const double a);
    const Vector operator*(const double a, const Vector& x);

    const SparseFeature operator*(const SparseFeature& f, const double a);
    const SparseFeature operator*(const double a, const SparseFeature& f);
    const DenseFeature operator*(const DenseFeature& f, const double a);
    const DenseFeature operator*(const double a, const DenseFeature& f);

    Vector& operator+=(Vector& x, const Vector &y);
    Vector& operator+=(Vector& x, const SparseFeature &f);
    Vector& operator+=(Vector& x, const DenseFeature &f);
    Vector& operator+=(Vector& x, const double a);
    Vector& operator-=(Vector& x, const Vector &y);
    Vector& operator-=(Vector& x, const SparseFeature &f);
    Vector& operator-=(Vector& x, const DenseFeature &f);
    Vector& operator-=(Vector& x, const double a);
    Vector& operator*=(Vector& x, const double a);
    bool operator== (const Vector& x, const Vector& y);
    bool operator!= (const Vector& x, const Vector& y);
    bool operator< (const Vector& x, const Vector& y);
    bool operator<= (const Vector& x, const Vector& y);
    bool operator> (const Vector& x, const Vector& y);
    bool operator>= (const Vector& x, const Vector& y);

    std::ostream& operator<<(std::ostream& os, const Vector& x);
    // ContinuousFunctions
    class ContinuousFunctions {
    protected:
        int n;                  // The number of convex functions added together, i.e if g(X) = \sum_{i = 1}^n f_i(x)
        int m;                 // Dimension of vectors or features (i.e. size of x in f(x))
    public:
        bool isSmooth;

        ContinuousFunctions(bool isSmooth);

        ContinuousFunctions(bool isSmooth, int m, int n);

        ContinuousFunctions(const ContinuousFunctions &c);         // copy constructor

        virtual ~ContinuousFunctions();

        virtual double eval(const Vector &x) const;                 // functionEval
        virtual Vector evalGradient(const Vector &x) const;                 // gradientEval
        virtual void
        eval(const Vector &x, double &f, Vector &gradient) const;                 // combined function and gradient eval
        virtual Vector
        evalStochasticGradient(const Vector &x, std::vector<int> &batch) const;                 // stochastic gradient
        virtual void evalStochastic(const Vector &x, double &f, Vector &g,
                                    std::vector<int> &miniBatch) const;                 // stochastic combined evaluation
        virtual Matrix evalHessian(const Vector &x) const;                      // hessianEval
        virtual void evalHessianVectorProduct(const Vector &x, const Vector &v,
                                              Vector &Hxv) const;                 // evaluate a product between a hessian and a vector
        double operator()(const Vector &x) const;

        int size() const;                 // number of features or dimension size (m)
        int length() const;                 // number of convex functions adding up (n)
    };
    // L2ProbitLoss
    template <class Feature>
    class L2ProbitLoss : public ContinuousFunctions {
    protected:
        std::vector<Feature>& features;                 // size of features is number of trainins examples (n)
        Vector& y;                 // size of y is number of training examples (n)
        double lambda;
        Vector &sum_msg_;
        int num_neighbor_;
        double rho_;
    public:
        L2ProbitLoss(int numFeatures, std::vector<Feature>& features, Vector& y, Vector &sum_msg, int num_neighbor,
                     double lambda, double rho);
        L2ProbitLoss(const L2ProbitLoss& c);         // copy constructor

        ~L2ProbitLoss();

        double eval(const Vector& x) const;                 // functionEval
        Vector evalGradient(const Vector& x) const;                 // gradientEval
        void eval(const Vector& x, double& f, Vector& gradient) const;                 // combined function and gradient eval
        Vector evalStochasticGradient(const Vector& x, std::vector<int>& miniBatch) const;                 // stochastic gradient
        void evalStochastic(const Vector& x, double& f, Vector& g, std::vector<int>& miniBatch) const;                 // stochastic evaluation
    };
    Vector gdNesterov(const ContinuousFunctions &c, const Vector &x0, double alpha = 1, const double gamma = 1e-4,
                      const int maxEval = 1000, const double TOL = 1e-3, bool resetAlpha = true,
                      bool useinputAlpha = false, int verbosity = 1);

}
// simpleallreduce.h
using namespace comlkit;
namespace spar {
    enum MessageType {
        kSimpleAllreduce1=0x10000,
        kSimpleAllreduce2,
        kScatterReduce,
        kAllGather,
        kmaxandmin,
        decencomm1,
        decencomm2
    };

    struct SumOperator {};

    struct MinOperator {};

    struct MaxOperator {};

    struct ProductOperator {};

    template<class O, class T>
    void Reduce(T &a, const T &b);

    template<>
    inline void Reduce<SumOperator>(double &a, const double &b) {
        a += b;
    }

    template<>
    inline void Reduce<SumOperator>(long long &a, const long long &b) {
        a += b;
    }

    template<>
    inline void Reduce<MinOperator>(double &a, const double &b) {
        if (b < a) {
            a = b;
        }
    }

    template<>
    inline void Reduce<MaxOperator>(double &a, const double &b) {
        if (b > a) {
            a = b;
        }
    }

    template<>
    inline void Reduce<ProductOperator>(double &a, const double &b) {
        a *= b;
    }

    template<>
    inline void Reduce<SumOperator>(float &a, const float &b) {
        a += b;
    }

    template<>
    inline void Reduce<MinOperator>(float &a, const float &b) {
        if (b < a) {
            a = b;
        }
    }

    template<>
    inline void Reduce<MaxOperator>(float &a, const float &b) {
        if (b > a) {
            a = b;
        }
    }

    template<>
    inline void Reduce<ProductOperator>(float &a, const float &b) {
        a *= b;
    }

    template<>
    inline void Reduce<SumOperator>(int &a, const int &b) {
        a += b;
    }

    template<>
    inline void Reduce<MinOperator>(int &a, const int &b) {
        if (b < a) {
            a = b;
        }
    }

    template<>
    inline void Reduce<MaxOperator>(int &a, const int &b) {
        if (b > a) {
            a = b;
        }
    }

    template<>
    inline void Reduce<ProductOperator>(int &a, const int &b) {
        a *= b;
    }

    template<>
    inline void Reduce<SumOperator>(long &a, const long &b) {
        a += b;
    }

    template<>
    inline void Reduce<MinOperator>(long &a, const long &b) {
        if (b < a) {
            a = b;
        }
    }

    template<>
    inline void Reduce<MaxOperator>(long &a, const long &b) {
        if (b > a) {
            a = b;
        }
    }

    template<>
    inline void Reduce<ProductOperator>(long &a, const long &b) {
        a *= b;
    }

    template<class T>
    inline int Send(const T *buf, int count, int dest, int tag, MPI_Comm comm);

    template<>
    inline int Send(const double *buf, int count, int dest, int tag, MPI_Comm comm) {
        return MPI_Send(buf, count, MPI_DOUBLE, dest, tag, comm);
    }

    template<>
    inline int Send(const float *buf, int count, int dest, int tag, MPI_Comm comm) {
        return MPI_Send(buf, count, MPI_FLOAT, dest, tag, comm);
    }

    template<>
    inline int Send(const int *buf, int count, int dest, int tag, MPI_Comm comm) {
        return MPI_Send(buf, count, MPI_INT, dest, tag, comm);
    }

    template<>
    inline int Send(const char *buf, int count, int dest, int tag, MPI_Comm comm) {
        return MPI_Send(buf, count, MPI_CHAR, dest, tag, comm);
    }

    template<>
    inline int Send(const short *buf, int count, int dest, int tag, MPI_Comm comm) {
        return MPI_Send(buf, count, MPI_SHORT, dest, tag, comm);
    }

    template<>
    inline int Send(const long *buf, int count, int dest, int tag, MPI_Comm comm) {
        return MPI_Send(buf, count, MPI_LONG, dest, tag, comm);
    }

    template<class T>
    inline int Isend(const T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);

    template<>
    inline int Isend(const double *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Isend(buf, count, MPI_DOUBLE, dest, tag, comm, request);
    }

    template<>
    inline int Isend(const uint16_t *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Isend(buf, count, MPI_UINT16_T, dest, tag, comm, request);
    }

    template<>
    inline int Isend(const float *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Isend(buf, count, MPI_FLOAT, dest, tag, comm, request);
    }

    template<>
    inline int Isend(const int *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Isend(buf, count, MPI_INT, dest, tag, comm, request);
    }

    template<>
    inline int Isend(const char *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Isend(buf, count, MPI_CHAR, dest, tag, comm, request);
    }

    template<>
    inline int Isend(const short *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Isend(buf, count, MPI_SHORT, dest, tag, comm, request);
    }

    template<>
    inline int Isend(const long *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Isend(buf, count, MPI_LONG, dest, tag, comm, request);
    }



    template<class T>
    inline int Recv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);

    template<>
    inline int Recv(double *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status) {
        return MPI_Recv(buf, count, MPI_DOUBLE, source, tag, comm, status);
    }

    template<>
    inline int Recv(float *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status) {
        return MPI_Recv(buf, count, MPI_FLOAT, source, tag, comm, status);
    }

    template<>
    inline int Recv(int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status) {
        return MPI_Recv(buf, count, MPI_INT, source, tag, comm, status);
    }

    template<>
    inline int Recv(char *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status) {
        return MPI_Recv(buf, count, MPI_CHAR, source, tag, comm, status);
    }

    template<>
    inline int Recv(short *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status) {
        return MPI_Recv(buf, count, MPI_SHORT, source, tag, comm, status);
    }

    template<>
    inline int Recv(long *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status) {
        return MPI_Recv(buf, count, MPI_LONG, source, tag, comm, status);
    }

    template<class T>
    inline int Irecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);

    template<>
    inline int Irecv(double *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Irecv(buf, count, MPI_DOUBLE, source, tag, comm, request);
    }


    template<>
    inline int Irecv(uint16_t *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Irecv(buf, count, MPI_UINT16_T, source, tag, comm, request);
    }

    template<>
    inline int Irecv(float *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Irecv(buf, count, MPI_FLOAT, source, tag, comm, request);
    }

    template<>
    inline int Irecv(int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Irecv(buf, count, MPI_INT, source, tag, comm, request);
    }

    template<>
    inline int Irecv(char *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Irecv(buf, count, MPI_CHAR, source, tag, comm, request);
    }

    template<>
    inline int Irecv(short *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Irecv(buf, count, MPI_SHORT, source, tag, comm, request);
    }

    template<>
    inline int Irecv(long *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request) {
        return MPI_Irecv(buf, count, MPI_LONG, source, tag, comm, request);
    }

//SimpleAllreduce主要在数据量特别少的时候使用
    template<class O, class T>
    void SimpleAllreduce(T *buffer, int count, int id, std::vector<int> &worker_list, MPI_Comm comm) {
        int worker_number = worker_list.size();
        if (worker_number == 1) return;
        T *recv_buffer = new T[count];
        int sender = 0, receiver = 1;
        for (int i = 0; i < worker_number - 1; ++i) {
            if (id == worker_list[sender]) {
                Send(buffer, count, worker_list[receiver], MessageType::kSimpleAllreduce1, comm);
            }
            if (id == worker_list[receiver]) {
                Recv(recv_buffer, count, worker_list[sender], MessageType::kSimpleAllreduce1, comm, MPI_STATUS_IGNORE);
                for (int j = 0; j < count; ++j) {
                    Reduce<O>(buffer[j], recv_buffer[j]);
                }
            }
            sender = (sender + 1) % worker_number;
            receiver = (receiver + 1) % worker_number;
        }

        for (int i = 0; i < worker_number - 1; ++i) {
            if (id == worker_list[sender]) {
                Send(buffer, count, worker_list[receiver], MessageType::kSimpleAllreduce2, comm);
            }
            if (id == worker_list[receiver]) {
                Recv(buffer, count, worker_list[sender], MessageType::kSimpleAllreduce2, comm, MPI_STATUS_IGNORE);
            }
            sender = (sender + 1) % worker_number;
            receiver = (receiver + 1) % worker_number;
        }
        delete []recv_buffer;
    }
}

class conf_util
{
public:
    conf_util();
    void parse(const string & conf_file);
    template<class T> T getItem(const string &item_name);
private:
    map<string, string> conf_items;
};

class args_t
{
public:
    args_t(int rank, int size);
    int myid;
    int procnum;
    string train_data_path;
    string test_data_path;
    string data_direction_;
    int Comm_method;
    int maxIteration;
    int nodesOfGroup;
    int Update_method;
    int Repeat_iter;
    // admm
    double rho;
    void get_args();
    void print_args();
};

// string_util.h
std::string &LeftTrim(std::string &s);

std::string &RightTrim(std::string &s);

std::string &Trim(std::string &s);
// Properties.h
class Properties {
public:
    Properties(const std::string &path);

    std::string GetString(const std::string &property_name);

    int GetInt(const std::string &property_name);

    double GetDouble(const std::string &property_name);

//    bool GetBool(const std::string &property_name);
//
//    bool HasProperty(const std::string &property_name);
//
//    void CheckProperty(const std::string &property_name);
//
//    void Print();

private:
    std::map<std::string, std::string> properties_;

    void ParseFromFile(const std::string &path);
};


////type_convert.h
template<typename Target, typename Source, bool Same>
class Converter {
public:
    static Target Convert(const Source &arg) {
        Target ret;
        std::stringstream ss;
        if (!(ss << arg && ss >> ret && ss.eof())) {
            //LOG(FATAL) << "类型转换失败";
        }
        return ret;
    }
};

template<typename Target, typename Source>
class Converter<Target, Source, true> {
public:
    static Target Convert(const Source &arg) {
        return arg;
    }
};

template<typename Source>
class Converter<std::string, Source, false> {
public:
    static std::string Convert(const Source &arg) {
        std::ostringstream ss;
        ss << arg;
        return ss.str();
    }
};

template<typename Target>
class Converter<Target, std::string, false> {
public:
    static Target Convert(const std::string &arg) {
        Target ret;
        std::istringstream ss(arg);
        if (!(ss >> ret && ss.eof())) {
            //LOG(FATAL) << "类型转换失败";
        }
        return ret;
    }
};

template<typename T1, typename T2>
struct IsSame {
    static const bool value = false;
};

template<typename T>
struct IsSame<T, T> {
    static const bool value = true;
};

template<typename Target, typename Source>
Target Convert(const Source &arg) {
    return Converter<Target, Source, IsSame<Target, Source>::value>::Convert(arg);
}

class GroupStrategy {
public:
    int repeatIter;

    GroupStrategy(int iternums);

    std::vector<int> exchangeElement(std::vector<int> data, int GroupNum, int Group1, int Group2, int part);

    std::vector<std::vector<int>> divideGroup(std::vector<int> nodes, int groupNums);

    int position(double *vec, int size, int index);

    std::vector<int> findFastNodes(double *time, std::vector<int> group, int node, int numsofGrup, int size);

    void changeGroup(std::vector<std::vector<int>> &data, int node, std::vector<int> fastVec, int numsOfgroup, int iter);

    void MasterNodes(int procnum, int nodesOfGroup, int DynamicGroup, int maxIteration, MPI_Comm comm);
};

class neighbors {
public:
    int neighborsNums;
    int *neighs;
    void setNeighbours(int nums,int *set);
    void clearNeighbours();
};

class ADMM {
public:

    ADMM(args_t *args, vector<struct SparseFeature> train_features, comlkit::Vector ytrain,
         vector<struct SparseFeature> test_features, comlkit::Vector ytest, int dimension, int optimizer, double beta, MPI_Comm comm);

    ~ADMM();

    void alpha_update(const comlkit::Vector &new_x, const comlkit::Vector &x_old, const comlkit::Vector &sumx);

    void group_train(clock_t start_time);

    double predict_comlkit(int method);

    double loss_value_comlkit(int method);

    void CreateGroup();

//    ofstream of;
    neighbors nears;
    double sum_cal_, sum_comm_;
    int quantify_part_, dynamic_group_, update_method_, sparse_comm_;
private:
    // ADMM algorithm
    int myid, procnum, dim_, data_number_;
    comlkit::Vector new_x_, sum_msg_, new_x_old_, new_alpha_, new_alpha_old_;
    double rho; // Penalty term parameter
    double beta_; // Inertial parameters
    int optimizer_;
    // Problem
    vector<struct SparseFeature> features_;
    Vector label_;
    Vector solution_;
    vector<struct SparseFeature> train_features_, test_features_;
    comlkit::Vector ytrain_, ytest_;
    int ntrian_, mtrian_, ntest_, mtest_;
    int maxIteration;
    int nodesOfGroup;
    // Group Strategy
    MPI_Comm SUBGRP_COMM_ODD_;
    MPI_Comm comm_;
    MPI_Comm SUBGRP_COMM_EVEN_;
    int *worker_ranks_;
};

void test_main(MPI_Comm comm);
void test_main2(MPI_Comm comm);
void test_main3(MPI_Comm comm);

//int sayhello(MPI_Comm comm);


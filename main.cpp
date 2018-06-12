#include <QCoreApplication>
#include <QDebug>
#include <QtMath>
#include <QFile>
#include <vector>
#include <omp.h>
#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <algorithm>
#include <x86intrin.h>

#define AVX
//#define SSE
//#define SEQ

class Matrix{
public:
    double sum(){
        double s = 0;
        for(auto vec : m_data){
            for (auto elem : vec){
                s += elem;
            }
        }
        return s;
    }
    Matrix(){}
    Matrix(int n, int m = - 1){
        m_data.clear();
        int secDim;
        if (m <= 0)
            secDim = n;
        else
            secDim = m;
        for (int i = 0; i < n; ++i){
            m_data.push_back(std::vector<float>{});
            for (int j = 0; j < secDim; ++j){
                m_data.at(i).push_back(0);
            }
        }
    }
    void initMatrix(){
        int n = 877;
        int mmin = 1;
        int mmax = 1000;
        std::hash<int> hf;
        for (int i = 0; i < n; ++i){
            m_data.push_back(std::vector<float>{});
            for (int j = 0; j < n; ++j){
                srand(hf(j * i));                
                m_data.at(i).push_back(rand()%(mmax - mmin +1) + mmin /* 5*/);
            }
        }
    }
    float& at(int i, int y){
        return m_data[i][y];
    }
    std::vector<std::vector<float> > m_data;

    //std::pair<int, int> len = {9,9};
    Matrix* Identity(int N){
        Matrix* E = new Matrix(N);
        for (int i = 0; i < N; ++i){
            E->at(i,i) = 1;
        }
        return E;
    }


    Matrix givens(Matrix &B){
        double EPS = 0.00000001;
        int n = B.m_data.size();
        double c = 0;
        double s = 0;
        double t = 0;
        Matrix* Q = this->Identity(n);
        Matrix* G = this->Identity(n);
        std::vector<std::vector<float> > Tr;

        for (int z = 0; z < n - 1; z++){
            Tr.push_back(std::vector<float>{});
        }
        for (int i = 0; i < n - 1; i++){         
            auto tmp_val = abs((int)B.at(i,i));
            if (tmp_val < EPS){
                c = 1;
                s = 0;
            }
            else{
                t = B.at(i + 1, i) / B.at(i,i);
                c = 1 / sqrt(1 + t * t);
                s = t * c;
            }
                Tr[i].push_back(c);
                Tr[i].push_back(s);
#ifdef SSE
                __m128 trj0 = _mm_set1_ps(Tr[i][0]);
                __m128 trj1 = _mm_set1_ps(Tr[i][1]);
#endif
#ifdef AVX
                __m256 trj0 = _mm256_set1_ps(Tr[i][0]);
                __m256 trj1 = _mm256_set1_ps(Tr[i][1]);
#endif

            for (int j = i; j < n; j++){
                float X, Y;
                X = B.at(i, j);//0
                Y = B.at(i+1, j);//3
                B.at(i,j) = c * X + s * Y; //0 * 0 = 0
                B.at(i+1, j) = c * Y - s * X; //1 * 3 - 0
            }

            this->prettyPrint();
            for (int j = 0; j < n - 1; j++){
                int avxCycles = n/8;
                int leftoverElems = n%8;
                auto givensHelper = [&](Matrix &B, int k, int j, QString id){
                    auto trj0 = Tr[i][0];
                    auto trj1 = Tr[i][1];                    
                    float X, Y;
                    X = B.at(k, j); //0
                    Y = B.at(k, j + 1); //1
                    B.at(k, j) = trj0 * X + trj1 * Y; //0 + 1 * Tr[j][1]
                    B.at(k, j + 1) = - trj1 * X + trj0 * Y;
                    //qDebug() << "k: " << k << " j: " << j;
                };

                for (int k = 0; k < avxCycles * 8; k+=8){                    
#ifdef SSE
                __m128 xVec = _mm_set_ps(B.at(k,j),       B.at(k + 1,j),
                                                               B.at(k + 2,j), B.at(k + 3,j));
                __m128 yVec = _mm_set_ps(B.at(k,j + 1),       B.at(k + 1,j + 1),
                                                               B.at(k + 2,j + 1), B.at(k + 3,j + 1));

                __m128 b_kj = _mm_add_ps(_mm_mul_ps(trj0, xVec),   _mm_mul_ps(trj1, yVec) );
                __m128 b_k_plusone = _mm_add_ps(
                                                  (_mm_mul_ps( _mm_mul_ps(trj1, _mm_set1_ps(-1)) , xVec)), // -trj1 * X
                                                  (_mm_mul_ps(trj0, yVec)) ) ; //trj0 * Y

                for (int ndx = 0; ndx <4; ndx++){
                    float f;
                    _MM_EXTRACT_FLOAT(f, b_kj, ndx); //TODO - CHECK LATER

                }
///***               float *res_bkj = (float*) & b_kj;
///***                float *res_bkplusone = (float*) & b_k_plusone;
//                std::vector<float> res_vec_bkj;
//                std::vector<float> res_vec_bkj_plusone;
//                for (int ndx = 3; ndx >=0; ndx--){
//                    //res_vec_bkj.push_back(res_bkj[ndx]);
//                    //res_vec_bkj_plusone.push_back(res_bkplusone[ndx]);
//                }
                ///WORKING CODE BELOW, COMMENTED FOR TEST
                int ndx2 = 0;
//                for (int ndx = 3; ndx >=0; ndx--){
//                    B.at(k+ndx2,j) = res_bkj[ndx];
//                    B.at(k+ndx2,j + 1) = res_bkplusone[ndx];
//                    ndx2++;
//                }

                xVec = _mm_set_ps(  B.at(k+4,j),       B.at(k + 5,j),
                                                      B.at(k + 6,j), B.at(k + 7,j));

                yVec = _mm_set_ps(  B.at(k,j + 4),       B.at(k + 5,j + 1),
                                                      B.at(k + 6,j + 1), B.at(k + 7,j + 1));

                b_kj = _mm_add_ps(_mm_mul_ps(trj0, xVec),   _mm_mul_ps(trj1, yVec) );
                b_k_plusone = _mm_add_ps(
                                                  (_mm_mul_ps( _mm_mul_ps(trj1, _mm_set1_ps(-1)) , xVec)), // -trj1 * X
                                                  (_mm_mul_ps(trj0, yVec)) ) ; //trj0 * Y
                float *res_bkj2 = (float*) & b_kj;
                float *res_bkplusone2 = (float*) & b_k_plusone;

                for (int ndx = 3; ndx >=0; ndx--){
                    B.at(k+ndx2,j) = res_bkj2[ndx];
                    B.at(k+ndx2,j + 1) = res_bkplusone2[ndx];
                    ndx2++;
                }
//                for (int ndx = 3; ndx >=0; ndx--){
//                    res_vec_bkj.push_back(res_bkj2[ndx]);
//                    res_vec_bkj_plusone.push_back(res_bkplusone2[ndx]);
//                }
//                for (int ndx = 0; ndx < 8; ndx++){
//                    B.at(k + ndx, j) = res_vec_bkj[ndx];
//                    B.at(k + ndx, j + 1) = res_vec_bkj_plusone[ndx];
//                }

#endif
#ifdef AVX
            __m256 xVec = _mm256_set_ps(B.at(k,j),       B.at(k + 1,j),
                                                           B.at(k + 2,j), B.at(k + 3,j),
                                                           B.at(k + 4,j), B.at(k + 5,j),
                                                           B.at(k + 6,j), B.at(k + 7,j));

            __m256 yVec = _mm256_set_ps(B.at(k,j + 1),       B.at(k + 1, j + 1),
                                                           B.at(k + 2,j + 1), B.at(k + 3,j + 1),
                                                           B.at(k + 4,j + 1), B.at(k + 5,j + 1),
                                                           B.at(k + 6,j + 1), B.at(k + 7,j + 1));
          __m256 b_kj = _mm256_add_ps(_mm256_mul_ps(trj0, xVec),   _mm256_mul_ps(trj1, yVec) );
          __m256 b_k_plusone = _mm256_add_ps(
                                            (_mm256_mul_ps( _mm256_mul_ps(trj1, _mm256_set1_ps(-1)) , xVec)), // -trj1 * X
                                            (_mm256_mul_ps(trj0, yVec)) ) ; //trj0 * Y          
          //float *res_bkj = (float*) & b_kj;
          float res_bkj[8] = {0,0,0,0,0,0,0,0};
          //float *res_bkj = new float[8];
          _mm256_storeu_ps(res_bkj, b_kj);
          //float *res_bkplusone = (float*) & b_k_plusone;
          float res_bkplusone[8] = {0,0,0,0,0,0,0,0};
          //float *res_bkplusone = new float[8];
          _mm256_storeu_ps(res_bkplusone, b_k_plusone);
//          qDebug() << res_bkj << "---" <<  res_bkj;
//          int mmm=0;
//          std::cin >>mmm;

          //std::vector<double> d1VecSIMD;
          //std::vector<double> d2VecSIMD;
          int ndx2 = 0;
          for (int ndx = 7; ndx >= 0; ndx--){
              B.at(k + ndx2,j) = res_bkj[ndx];
              B.at(k + ndx2, j+1) = res_bkplusone[ndx];
              ndx2++;
              //d1VecSIMD.push_back(res_bkj[ndx]);
              //d2VecSIMD.push_back(res_bkplusone[ndx]);
          }
          //delete[] res_bkj;
          //delete[] res_bkplusone;
//          qDebug() << "d 1 SIMD: " << d1VecSIMD;
//          qDebug() << "d 2 SIMD: " << d2VecSIMD;


#endif

#ifdef  SEQ
                    givensHelper(B,k,j, "avx planned loop");
                    givensHelper(B,k + 1,j, "avx planned loop");
                    givensHelper(B,k + 2,j, "avx planned loop");
                    givensHelper(B,k + 3,j, "avx planned loop");
                    givensHelper(B,k + 4,j, "avx planned loop");
                    givensHelper(B,k + 5,j, "avx planned loop");
                    givensHelper(B,k + 6,j, "avx planned loop");
                    givensHelper(B,k + 7,j, "avx planned loop"); 
//                    std::vector<double> d1Vec;
//                    std::vector<double> d2Vec;
//                    for (int ndx = 0; ndx < 8; ndx++){
//                        d1Vec.push_back( B.at(k + ndx, j) );
//                        d2Vec.push_back( B.at(k + ndx, j + 1) );
//                    }
//                    qDebug() << "d1 seq vec: " << d1Vec;
//                    qDebug() << "d2 seq vec: " << d2Vec;
#endif
                }                

                for (int qq = avxCycles * 8; qq < n; qq++){
                    givensHelper(B,qq,j, " non avx planned loop");
                }

            }
        }//main for

        return B;
    }//Givens fn

    void prettyPrint(bool dbg = false){
        if(!dbg)
            return;
        for (std::vector<float> row : m_data){
             std::cout << std::endl << "----------------------------------------" << std::endl;
            for (float d : row){
                std::cout << "      " << d;
            }

        }
        std::cout << std::endl;

    }
};

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    Matrix m;
    m.initMatrix();
    auto start_time = std::chrono::high_resolution_clock::now();
    m.givens(m);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
#ifdef AVX
    std::cout << "AVX runtime:" <<std::endl;
#endif
#ifdef SEQ
    std::cout << "SEQ runtime:" <<std::endl;
#endif
#ifdef SSE
    std::cout << "SSE bit runtime:" <<std::endl;
#endif
    std::cout << "time: " << duration << " ms" << std::endl;

    //std::vector<float> test = m.m_data.at(1);
    //qDebug() << test;
    //m.prettyPrint(true);

    return a.exec();
}

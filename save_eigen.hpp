#ifndef SAVE_EIGEN_HPP
#define SAVE_EIGEN_HPP

#include <cstdio>
#include <cstdint>
#include <Eigen/Core>

inline void save_eigen(const Eigen::VectorXd &x, const char *name)
{
    std::FILE *ofile = fopen(name, "wb");
    int64_t sz = x.rows();

    fwrite(&sz, sizeof(sz), 1, ofile);
    fwrite(x.data(), sizeof(double), x.rows(), ofile);
    fclose(ofile);
}

#endif // SAVE_EIGEN_HPP

#pragma once

#include <vector>

namespace NuTo
{
namespace TimeIntegration
{
template <typename Tstate>
class ExplicitNystroemNoVelocity
{

public:
    //! @brief Initialization with method specific parameters (butcher tableau)
    ExplicitNystroemNoVelocity(std::vector<std::vector<double>> aa2, std::vector<double> bb1, std::vector<double> bb2,
                               std::vector<double> cc)
        : a2(aa2)
        , b1(bb1)
        , b2(bb2)
        , c(cc)
    {
    }

    //! @brief Performs one Nystroem step
    //! @param f A functor that returns the right hand side of
    //! the differential equation y'' = f(y,t)
    //!
    //! The signature of its call operator must be:
    //! operator()(const Tstate& w, Tstate& d2wdt2, double t)
    //! The return value is stored in dwdt
    //!
    //! @param w0 initial value
    //! @param v0 initial velocity
    //! @param t0 start time
    //! @param h step size (t-t0)
    //! @return value after one Nystroem step
    template <typename F>
    std::pair<Tstate, Tstate> DoStep(F f, Tstate w0, Tstate v0, double t0, double h)
    {
        if (k.empty())
        {
            k.resize(c.size());
        }
        for (int i=0; i<k.size(); i++)
        {
            k[i] = w0;
        }
        Tstate resultW = w0 + h * v0;
        Tstate resultV = v0;

        for (std::size_t i = 0; i < c.size(); i++)
        {
            double t = t0 + c[i] * h;
            Tstate wni = w0 + h * v0 * c[i];
            for (std::size_t j = 0; j < i; j++)
            {
                if (a2[i][j] != 0)
                    wni += h * h * a2[i][j] * k[j];
            }
            f(wni, k[i], t);
            resultW += h * h * b2[i] * k[i];
            resultV += h * b1[i] * k[i];
        }
        return std::make_pair(resultW, resultV);
    }

protected:
    std::vector<std::vector<double>> a2;
    std::vector<double> b1;
    std::vector<double> b2;
    std::vector<double> c;
    std::vector<Tstate> k;
};
}
}






















/*
 * Copyright 2020 The University of Texas at Austin.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef C0_TRIANGLES_HPP
#define C0_TRIANGLES_HPP

#include "Galerkin/Galerkin.hpp"
using Galerkin::Elements::derive_shape_functions;
using Galerkin::Elements::evaluate_at;
using Galerkin::Metanomials::Powers;
using Galerkin::Rationals::rational;

namespace fem
{

namespace c0
{

template <int PolynomialOrder>
struct C0Triangle;

} // namespace c0

} //namespace fem

template <int PolynomialOrder>
struct Galerkin::DefaultIntegrationOrder<fem::c0::C0Triangle<PolynomialOrder>> :
    public Galerkin::IntegrationOrder<2*PolynomialOrder>
{};

namespace fem {

namespace c0 {

template <int PolynomialOrder>
struct C0TriangleBasis{};

template <>
struct C0TriangleBasis<1> :
    public decltype(derive_shape_functions(
        Galerkin::Elements::make_form(
            Powers<1, 0>{}, Powers<0, 1>{}, Powers<0, 0>{}),
        Galerkin::make_list(evaluate_at(rational<-1>, rational<-1>),
                            evaluate_at(rational<-1>, rational<1>),
                            evaluate_at(rational<1>, rational<-1>))
    ))
{};

template <>
struct C0TriangleBasis<2> :
    public decltype(derive_shape_functions(
        Galerkin::Elements::make_form(
            Powers<2, 0>{}, Powers<0, 2>{}, Powers<1, 1>{},
            Powers<1, 0>{}, Powers<0, 1>{}, Powers<0, 0>{}),
        Galerkin::make_list(evaluate_at(rational<-1>, rational<-1>),
                            evaluate_at(rational<-1>, rational<1>),
                            evaluate_at(rational<1>, rational<-1>),
                            evaluate_at(rational<-1>, rational<0>),
                            evaluate_at(rational<0>, rational<0>),
                            evaluate_at(rational<0>, rational<-1>))))
{};

template <>
struct C0TriangleBasis<3> :
    public decltype(derive_shape_functions(
        Galerkin::Elements::make_form(
            Powers<3, 0>{}, Powers<0, 3>{}, Powers<2, 1>{},
            Powers<1, 2>{}, Powers<2, 0>{}, Powers<0, 2>{},
            Powers<1, 1>{}, Powers<1, 0>{}, Powers<0, 1>{},
            Powers<0, 0>{}),
        Galerkin::make_list(evaluate_at(rational<-1>, rational<-1>),
                            evaluate_at(rational<-1>, rational<1>),
                            evaluate_at(rational<1>, rational<-1>),
                            evaluate_at(rational<-1>, -rational<1, 3>),
                            evaluate_at(rational<-1>, rational<1, 3>),
                            evaluate_at(-rational<1, 3>, rational<1, 3>),
                            evaluate_at(rational<1, 3>, -rational<1, 3>),
                            evaluate_at(rational<1, 3>, -rational<1>),
                            evaluate_at(-rational<1, 3>, -rational<1>),
                            evaluate_at(-rational<1, 2>, -rational<1, 2>))))
{};

template <>
struct C0TriangleBasis<4> :
    public decltype(derive_shape_functions(
        Galerkin::Elements::make_form(
            Powers<4, 0>{}, Powers<0, 4>{}, Powers<3, 1>{}, Powers<1, 3>{},
            Powers<2, 2>{}, Powers<3, 0>{}, Powers<0, 3>{}, Powers<2, 1>{},
            Powers<1, 2>{}, Powers<2, 0>{}, Powers<0, 2>{}, Powers<1, 1>{},
            Powers<1, 0>{}, Powers<0, 1>{}, Powers<0, 0>{}),
        Galerkin::make_list(evaluate_at(rational<-1>, rational<-1>),
                            evaluate_at(rational<-1>, rational<1>),
                            evaluate_at(rational<1>, rational<-1>),
                            evaluate_at(rational<-1>, -rational<1, 2>),
                            evaluate_at(rational<-1>, rational<0>),
                            evaluate_at(rational<-1>, rational<1, 2>),
                            evaluate_at(-rational<1, 2>, rational<1, 2>),
                            evaluate_at(rational<0>, rational<0>),
                            evaluate_at(rational<1, 2>, -rational<1, 2>),
                            evaluate_at(rational<1, 2>, rational<-1>),
                            evaluate_at(rational<0>, rational<-1>),
                            evaluate_at(rational<-1, 2>, rational<-1>),
                            evaluate_at(-rational<1, 2>, -rational<1, 2>),
                            evaluate_at(-rational<1, 2>, rational<0>),
                            evaluate_at(rational<0>, -rational<1, 2>))
    ))
{};

template <int PolynomialOrder>
struct C0Triangle : public Galerkin::Elements::ElementBase<C0Triangle<PolynomialOrder>>
{
    constexpr static auto basis = C0TriangleBasis<PolynomialOrder>{};

    const auto &coordinate_map() const noexcept
    {
        return m_map;
    }

    template <class P>
    C0Triangle(const P &a, const P &b, const P &c) noexcept :
        m_map(a, b, c)
    {}

private:
    Galerkin::Transforms::TriangleTransform<double> m_map;
};

} // namespace c0

} // namespace fem

#endif // C0_TRIANGLES_HPP

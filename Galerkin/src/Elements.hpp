/*
 * Copyright (c) 2019, The University of Texas at Austin & Sean McBane
 * This file is part of the Galerkin library; Galerkin is copyright under the
 * terms of the MIT license. Please see the top-level COPYRIGHT file for details.
 */

#ifndef ELEMENTS_HPP
#define ELEMENTS_HPP

/*!
 * @file Elements.hpp
 * @brief Basic functionality related to defining element types.
 */

#include "MetaLinAlg.hpp"
#include "Metanomials.hpp"
#include "utils.hpp"

#include <tuple>

namespace Galerkin
{

namespace Elements
{

/*!
 * @brief Represents the form of a polynomial shape function.
 * 
 * The `ShapeFunctionForm` is just an empty variadic template struct, where the
 * template parameters are types of `Powers` classes representing the terms of a
 * metanomial. For example, the following:
 * 
 *     constexpr ShapeFunctionForm<Powers<0, 0>, Powers<0, 1>, Powers<1, 0>, Powers<1, 1>> form{};
 * 
 * represents the form of a bilinear shape function `phi(x, y) = a*x*y + b*x + c*y + d`.
 */
template <class... Powers>
struct ShapeFunctionForm : public typeconst_list<Powers...>
{
};

// Helper to make sure arguments to make_form are all Powers objects.
namespace
{

template <class P, class... Ps>
constexpr bool check_powers() noexcept
{
    if constexpr (sizeof...(Ps) == 0)
    {
        return Metanomials::is_powers<P>;
    }
    else
    {
        return Metanomials::is_powers<P> && check_powers<Ps...>();
    }
}

} // namespace

/*!
 * @brief Convert a `typeconst_list` to a `ShapeFunctionForm`
 *
 * This function checks that all of the types `Ps...` are instantiations of
 * `Metanomials::Powers`, then returns a `ShapeFunctionForm` with the same
 * powers.
 */
template <class... Ps>
constexpr auto to_form(typeconst_list<Ps...>) noexcept
{
    static_assert(check_powers<Ps...>(), "All arguments to to_form should be 'Powers' objects");
    return ShapeFunctionForm<Ps...>();
}

/*!
 * @brief Construct a `ShapeFunctionForm` from variadic list of `Metanomials::Powers` objects
 * 
 * The initialized form has duplicate `Powers` objects merged into one and terms
 * sorted in ascending order. Arguments are checked to make sure that they are
 * actually `Powers`.
 */
template <class... Powers>
constexpr auto make_form(Powers...) noexcept
{
    static_assert(check_powers<Powers...>(), "All arguments to make_form should be 'Powers' objects");
    return to_form(ShapeFunctionForm<Powers...>().sorted().unique());
}

/*!
 * @brief Combine two `Powers` objects
 *
 * This is the analogue of `std::tuple_cat` or `typeconst_list::append`.
 * Combine the powers in two `Powers` objects into one `Powers` objects;
 * e.g. `concatenate_powers(Powers<1>{}, Powers<1>{}) == Powers<1, 1>{}`.
 */
template <auto... Is, auto... Js>
constexpr auto concatenate_powers(Metanomials::Powers<Is...>, Metanomials::Powers<Js...>) noexcept
{
    return Metanomials::Powers<Is..., Js...>{};
}

/*!
 * @brief Returns a `ShapeFunctionForm` with combination of powers up to given maxima.
 * 
 * This function is a utility to construct the common form for shape functions
 * that consists of all terms that are at most order N in a given term.
 * For example, `powers_up_to(intgr_constant<1>, intgr_constant<1>)` gives the
 * form of a bilinear shape function `f(x, y) = axy + bx + cy + d`.
 */
template <auto I, auto... Is>
constexpr auto powers_up_to(std::integral_constant<decltype(I), I>, 
                            std::integral_constant<decltype(Is), Is>...) noexcept
{
    static_assert(I >= 0);
    if constexpr (sizeof...(Is) == 0)
    {
        constexpr auto lst = static_reduce<0, I+1, 1>
        (
            [](auto i) { return i; },
            typeconst_list<>{},
            [](auto l, auto i)
            {
                return l.append(typeconst_list<Metanomials::Powers<i()>>{});
            }
        );
        return to_form(lst);
    }
    else
    {
        constexpr auto lst = powers_up_to(intgr_constant<I>);
        constexpr auto tails = powers_up_to(intgr_constant<Is>...);
        constexpr auto power_list = static_reduce<0, lst.count, 1>(
            [=](auto i)
            {
                constexpr auto index1 = i();
                return static_reduce<0, tails.count, 1>(
                    [=](auto j)
                    {
                        return concatenate_powers(get<index1>(lst), get<j()>(tails));
                    },
                    typeconst_list<>{},
                    [](auto l1, auto pow) { return l1.append(typeconst_list<decltype(pow)>{}); }
                );
            },
            typeconst_list<>{},
            [](auto l1, auto l2) { return l1.append(l2); }
        );
        return to_form(power_list);
    }
}

// Here ends boilerplate for the DSL and begins the actual implementation of
// deriving shape functions.
namespace
{

template <class... Powers, class... Constraints>
constexpr auto build_terms_matrix(ShapeFunctionForm<Powers...>, typeconst_list<Constraints...>) noexcept
{
    return typeconst_list<Constraints...>().map(
        [](auto constraint)
        {
            return make_list(constraint(Powers())...);
        }
    );
}

template <class... Coeffs, class... Powers>
constexpr auto multiply_coeffs(typeconst_list<Coeffs...>, ShapeFunctionForm<Powers...>) noexcept
{
    return Metanomials::metanomial(Metanomials::term(Coeffs(), Powers())...);
}

} // namespace

/*!
 * @brief Given degrees of freedom for an element, derive shape functions.
 * 
 * This function derives polynomial shape functions on an element given the
 * degrees of freedom on the element in a functional form. The first argument is
 * a `ShapeFunctionForm` specifying the powers of the metanomial form, and the
 * second is a `typeconst_list` of constraints. These take the form of a function
 * object accepting a `Powers` object (see `Metanomials.hpp` for interface) and
 * returning a number as a `Rationals::Rational`. Functors for the most common cases are
 * provided - see `evaluate_at` and `partial_at`.
 *
 * @see evaluate_at
 * @see partial_at
 */
template <class... Powers, class... Constraints>
constexpr auto derive_shape_functions(ShapeFunctionForm<Powers...>, typeconst_list<Constraints...>) noexcept
{
    static_assert(sizeof...(Powers) == sizeof...(Constraints), "Ill-posed system to derive shape functions");
    constexpr auto terms_matrix = build_terms_matrix(
        ShapeFunctionForm<Powers...>(), typeconst_list<Constraints...>());

    return static_reduce<0, sizeof...(Constraints), 1>(
        [=](auto I) {
            constexpr auto coeffs = MetaLinAlg::linear_solve(terms_matrix,
                                                             MetaLinAlg::canonical<I(), sizeof...(Constraints)>());
            return multiply_coeffs(coeffs, ShapeFunctionForm<Powers...>());
        },
        typeconst_list<>(),
        [](auto L, auto x) { return L.append(make_list(x)); });
}

/*!
 * @brief A constraint functor for use in `derive_shape_functions`.
 *
 * An instance of this class, when a member of the list of constraints given to
 * `derive_shape_functions`, indicates that one of the degrees of freedom
 * constraining the system is the value of a function at the point
 * `(Ns...)` in R^n (where `n == sizeof...(Ns)`). There are no data members;
 * the point at which to evaluate must be specified using either
 * a `Rationals::Rational` or `std::integral_constant`. Construct an instance
 * of this functor using the helper function `evaluate_at`.
 *
 * @see derive_shape_functions
 * @see evaluate_at
 */
template <class... Ns>
struct EvaluateAt
{
    template <class Powers>
    constexpr auto operator()(Powers) const noexcept
    {
        return Metanomials::raise(std::tuple(Ns()...), Powers());
    }
};

// Helper function to check types of coordinate variables.
namespace
{
    
template <class Coord, class... Coords>
constexpr bool check_coords() noexcept
{
    constexpr bool is_coord = Rationals::is_rational<Coord> || is_intgr_constant<Coord>;
    if constexpr (sizeof...(Coords) == 0)
    {
        return is_coord;
    }
    else
    {
        return is_coord && check_coords<Coords...>();
    }
}

} // namespace

/*!
 * @brief Construct an instance of EvaluateAt given arguments by value
 *
 * Rather than the unwieldy syntax `EvaluateAt<decltype(Ns)...>{}`, use this
 * helper function to construct the instance when given `Ns` by value instead
 * of as types.
 *
 * @see EvaluateAt
 */
template <class... Ns>
constexpr auto evaluate_at(Ns...) noexcept
{
    static_assert(check_coords<Ns...>(), "All coordinates should be rationals or integral_constants");
    return EvaluateAt<Ns...>();
}

/*!
 * @brief Functor representing a partial derivative value as DOF
 *
 * This class is intended for use as a constraint in `derive_shape_functions`.
 * The indices `(I, Is...)` represent the variable which should be
 * differentiated with respect to, and the point at which to take the partial
 * derivative is a `typeconst_list` encoded in the type `CoordList`.
 *
 * For example, a degree of freedom which is the value of the cross derivative
 * `\frac{\partial^2 f}{\partial x \partial y}` at the origin would be encoded
 * as the type `PartialAt<typeconst_list<Rationals::Rational<0, 1>, Rationals::Rational<0, 1>>, 0, 1>`.
 *
 * In practice, construct an instance using `partial_at`.
 *
 * @see partial_at
 * @see derive_shape_functions
 */
template <class CoordList, auto I, auto... Is>
class PartialAt
{
public:
    template <class Powers>
    constexpr auto operator()(Powers) const noexcept
    {
        constexpr auto t = take_partials<I, Is...>(Powers());
        return t(instantiate_tuple(CoordList()));
    }

private:

    template <auto J, auto... Js, class Powers>
    static constexpr auto take_partials(Powers) noexcept
    {
        constexpr auto first_partial = partial<J>(
            Metanomials::term(Rationals::rational<1>, Powers())
        );

        if constexpr (sizeof...(Js) == 0)
        {
            return first_partial;
        }
        else
        {
            return first_partial.coeff() * take_partials<Js...>(first_partial.powers());
        }
    }

    template <class... Coords>
    static constexpr auto instantiate_tuple(typeconst_list<Coords...>) noexcept
    {
        return std::tuple(Coords()...);
    }
};

/*!
 * @brief Helper to construct `PartialAt` instance correctly.
 *
 * Use `partial_at<I, Is...>(coords...)` to construct a `PartialAt` instance
 * representing a constraint on the value of the partial derivative with
 * respect to the variables indexed by `(I, Is...)` at the point in `R^n`
 * given by `(coords...)`. For example, a constraint on the cross derivative
 * `\frac{\partial^2 f}{\partial x \partial y}` at the origin can be
 * constructed using `partial_at<0, 1>(Rationals::rational<0>, Rationals::rational<0>)`.
 */
template <auto I, auto... Is, class... Coords>
constexpr auto partial_at(Coords...) noexcept
{
    static_assert(check_coords<Coords...>(), "All coordinates should be rationals or integral_constants");
    return PartialAt<typeconst_list<Coords...>, I, Is...>();
}

/********************************************************************************
 * Test derivation of shape functions given "control points" and a form for the
 * shape function.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

using namespace Metanomials;
using namespace Rationals;

TEST_CASE("[Galerkin::Elements] Test powers_up_to")
{
    constexpr auto powers = powers_up_to(intgr_constant<1>, intgr_constant<1>);
    REQUIRE(std::is_same_v<decltype(powers), const ShapeFunctionForm<Powers<0, 0>, Powers<0, 1>, Powers<1, 0>, Powers<1, 1>>>);
}

TEST_CASE("[Galerkin::Elements] Deriving one-dimensional shape functions")
{
    SUBCASE("Test for a first order element")
    {
        // Shape function has the form ax + b.
        constexpr auto form = make_form(
            powers(intgr_constant<1>),
            powers(intgr_constant<0>));

        // Constraints are the function values at -1, 1.
        constexpr auto constraints = make_list(
            evaluate_at(rational<-1>),
            evaluate_at(rational<1>));

        constexpr auto fns = derive_shape_functions(form, constraints);

        REQUIRE(get<0>(fns) ==
                metanomial(
                    term(-rational<1, 2>, powers(intgr_constant<1>)),
                    term(rational<1, 2>, powers(intgr_constant<0>))));
        REQUIRE(get<1>(fns) ==
                metanomial(
                    term(rational<1, 2>, powers(intgr_constant<1>)),
                    term(rational<1, 2>, powers(intgr_constant<0>))));
    }

    SUBCASE("Test for a second order element")
    {
        constexpr auto form = make_form(
            powers(intgr_constant<2>),
            powers(intgr_constant<1>),
            powers(intgr_constant<0>));

        constexpr auto constraints = make_list(
            evaluate_at(rational<-1>),
            evaluate_at(rational<0>),
            evaluate_at(rational<1>));
        
        constexpr auto fns = derive_shape_functions(form, constraints);

        REQUIRE(get<0>(fns) ==
            metanomial(
                term(rational<1, 2>, powers(intgr_constant<2>)),
                term(-rational<1, 2>, powers(intgr_constant<1>))
            ));

        REQUIRE(get<1>(fns) ==
            metanomial(
                term(-rational<1>, powers(intgr_constant<2>)),
                term(rational<1>, powers(intgr_constant<0>))
            ));

        REQUIRE(get<2>(fns) ==
            metanomial(
                term(rational<1, 2>, powers(intgr_constant<2>)),
                term(rational<1, 2>, powers(intgr_constant<1>))
            ));
    }

    SUBCASE("Test for a 3rd-order element with derivative DOF's")
    {
        constexpr auto form = make_form(
            powers(intgr_constant<3>),
            powers(intgr_constant<2>),
            powers(intgr_constant<1>),
            powers(intgr_constant<0>));

        constexpr auto constraints = make_list(
            evaluate_at(rational<-1>),
            evaluate_at(rational<1>),
            partial_at<0>(rational<-1>),
            partial_at<0>(rational<1>));

        constexpr auto fns = derive_shape_functions(form, constraints);

        REQUIRE(get<0>(fns) ==
            metanomial(
                term(rational<1, 4>, powers(intgr_constant<3>)),
                term(-rational<3, 4>, powers(intgr_constant<1>)),
                term(rational<1, 2>, powers(intgr_constant<0>))
            ));

        REQUIRE(get<1>(fns) ==
            metanomial(
                term(-rational<1, 4>, powers(intgr_constant<3>)),
                term(rational<3, 4>, powers(intgr_constant<1>)),
                term(rational<1, 2>, powers(intgr_constant<0>))
            ));

        REQUIRE(get<2>(fns) ==
            metanomial(
                term(rational<1, 4>, powers(intgr_constant<3>)),
                term(-rational<1, 4>, powers(intgr_constant<2>)),
                term(-rational<1, 4>, powers(intgr_constant<1>)),
                term(rational<1, 4>, powers(intgr_constant<0>))
            ));

        REQUIRE(get<3>(fns) ==
            metanomial(
                term(rational<1, 4>, powers(intgr_constant<3>)),
                term(rational<1, 4>, powers(intgr_constant<2>)),
                term(-rational<1, 4>, powers(intgr_constant<1>)),
                term(-rational<1, 4>, powers(intgr_constant<0>))
            ));
    }
}

TEST_CASE("[Galerkin::Elements] Deriving two-dimensional shape functions")
{
    SUBCASE("Test derivation of bilinear shape functions on a quadrilateral")
    {
        constexpr auto form = make_form(
            powers(intgr_constant<1>, intgr_constant<1>),
            powers(intgr_constant<1>, intgr_constant<0>),
            powers(intgr_constant<0>, intgr_constant<1>),
            powers(intgr_constant<0>, intgr_constant<0>)
        );

        constexpr auto constraints = make_list(
            evaluate_at(rational<-1>, rational<-1>),
            evaluate_at(rational<-1>, rational<1>),
            evaluate_at(rational<1>, rational<1>),
            evaluate_at(rational<1>, rational<-1>)
        );

        constexpr auto fns = derive_shape_functions(form, constraints);

        REQUIRE(get<0>(fns) ==
            metanomial(
                term(rational<1, 4>, powers(intgr_constant<1>, intgr_constant<1>)),
                term(-rational<1, 4>, powers(intgr_constant<1>, intgr_constant<0>)),
                term(-rational<1, 4>, powers(intgr_constant<0>, intgr_constant<1>)),
                term(rational<1, 4>, powers(intgr_constant<0>, intgr_constant<0>))
            ));

        REQUIRE(get<1>(fns) ==
            metanomial(
                term(-rational<1, 4>, powers(intgr_constant<1>, intgr_constant<1>)),
                term(-rational<1, 4>, powers(intgr_constant<1>, intgr_constant<0>)),
                term(rational<1, 4>, powers(intgr_constant<0>, intgr_constant<1>)),
                term(rational<1, 4>, powers(intgr_constant<0>, intgr_constant<0>))
            ));

        REQUIRE(get<2>(fns) ==
            metanomial(
                term(rational<1, 4>, powers(intgr_constant<1>, intgr_constant<1>)),
                term(rational<1, 4>, powers(intgr_constant<1>, intgr_constant<0>)),
                term(rational<1, 4>, powers(intgr_constant<0>, intgr_constant<1>)),
                term(rational<1, 4>, powers(intgr_constant<0>, intgr_constant<0>))
            ));

        REQUIRE(get<3>(fns) ==
            metanomial(
                term(-rational<1, 4>, powers(intgr_constant<1>, intgr_constant<1>)),
                term(rational<1, 4>, powers(intgr_constant<1>, intgr_constant<0>)),
                term(-rational<1, 4>, powers(intgr_constant<0>, intgr_constant<1>)),
                term(rational<1, 4>, powers(intgr_constant<0>, intgr_constant<0>))
            ));
    }
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 * End shape function API test.
 *******************************************************************************/
} // namespace Elements

} /* namespace Galerkin */

#endif /* ELEMENTS_HPP */

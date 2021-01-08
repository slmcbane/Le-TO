/*
 * Copyright (c) 2019, Sean McBane and The University of Texas at Austin.
 * This file is part of the Galerkin library; Galerkin is copyright under the
 * terms of the MIT license. Please see the top-level COPYRIGHT file for details.
 */

#ifndef METALINALG_HPP
#define METALINALG_HPP

/*!
 * @file MetaLinAlg.hpp
 * @brief A compile-time matrix facility intended to solve a linear system to
 * find coefficients of multinomial basis functions.
 */

#include "utils.hpp"
#include "Rationals.hpp"

#include <tuple>

namespace Galerkin
{

/*!
 * @brief Implements a solver for linear systems of equations at compile-time.
 */
namespace MetaLinAlg
{

/*!
 * @brief To store a row of a matrix we re-use the utility typeconst_list.
 * 
 * It is up to functions manipulating matrices to make sure that the types of
 * `Nums...` continue to make sense - they should all be rationals or integral
 * constants.
 */
template <class... Nums>
using MatrixRow = typeconst_list<Nums...>;

/*!
 * @brief Utility; construct a canonical basis vector.
 */
template <auto I, auto N>
constexpr auto canonical()
{
    static_assert(N >= 0);
    if constexpr (N == 0)
    {
        return make_list();
    }
    else if constexpr (I == 0)
    {
        return make_list(Rationals::rational<1>).append(canonical<-1, N-1>());
    }
    else
    {
        return make_list(Rationals::rational<0>).append(canonical<I-1, N-1>());
    }
}

/********************************************************************************
 * Test that canonical basis vector is correct.
 *******************************************************************************/
#ifdef DOCTEST_LIBRARY_INCLUDED

using namespace Galerkin::Rationals;

TEST_CASE("[Galerkin::MetaLinAlg] Canonical basis vectors")
{
    REQUIRE(canonical<0, 3>() == MatrixRow<Rational<1, 1>, Rational<0, 1>, Rational<0, 1>>());
    REQUIRE(canonical<1, 3>() == MatrixRow<Rational<0, 1>, Rational<1, 1>, Rational<0, 1>>());
    REQUIRE(canonical<2, 3>() == MatrixRow<Rational<0, 1>, Rational<0, 1>, Rational<1, 1>>());
}

#endif
/********************************************************************************
 * End test block
 *******************************************************************************/

/*!
 * @brief A `Matrix` is just a `typeconst_list` of `MatrixRow`s.
 */
template <class... Rows>
using Matrix = typeconst_list<Rows...>;

/*!
 * @brief Make a row from a list of rationals.
 */
template <auto N, auto D, class... Nums>
constexpr auto make_row(Rationals::Rational<N, D>, Nums...) noexcept
{
    constexpr auto head = MatrixRow<Rationals::Rational<N, D>>();
    if constexpr (sizeof...(Nums) == 0)
    {
        return head;
    }
    else
    {
        return head.append(make_row(Nums()...));
    }
}

/*!
 * @brief Make a matrix from a list of rows.
 * 
 * This function requires and checks via `static_assert` that the length of each
 * row is the same.
 */
template <class Row, class... Rows>
constexpr auto make_matrix(Row, Rows...) noexcept
{
    constexpr auto head = Matrix<Row>();
    if constexpr (sizeof...(Rows) == 0)
    {
        return head;
    }
    else
    {
        constexpr auto tail = make_matrix(Rows()...);
        static_assert(Row::count == tail.head().count,
                      "All matrix rows must have the same length");
        return head.append(tail);
    }
}

/*!
 * @brief Return the specified row from `mat`, using a 0-based index.
 */
template <auto I, class... Rows>
constexpr auto get_row(Matrix<Rows...> mat) noexcept
{
    return get<I>(mat);
}

/*!
 * @brief Return the specified element from `mat`, using a 0-based index.
 */
template <auto I, auto J, class... Rows>
constexpr auto get_elt(Matrix<Rows...> mat) noexcept
{
    return get<J>(get_row<I>(mat));
}

/********************************************************************************
 * Test making matrices and accessing elements using this API.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

using namespace Galerkin::Rationals;

TEST_CASE("[Galerkin::MetaLinAlg] Construct a MetaMatrix, access elements and rows")
{
    constexpr auto mat = Matrix<
        MatrixRow<Rational<1, 2>, Rational<1, 3>>,
        MatrixRow<Rational<1, 3>, Rational<1, 2>>>();

    constexpr auto mat2 = make_matrix(make_row(rational<1, 2>, rational<2, 6>),
                                      make_row(rational<3, 9>, rational<1, 2>));

    REQUIRE(mat == mat2);
    REQUIRE(get_row<0>(mat) == make_row(rational<1, 2>, rational<2, 6>));
    REQUIRE(get_elt<0, 0>(mat) == rational<1, 2>);
    REQUIRE(get_elt<1, 0>(mat) == rational<1, 3>);

    // Should throw a static_assert; mismatched row lengths.
    /* constexpr auto discard = make_matrix(
        MatrixRow<Rational<1, 2>, Rational<1, 3>>(),
        MatrixRow<Rational<1, 1>, Rational<1, 1>, Rational<1, 1>>()
     ); */

    // Shouldn't compile; rows do not all consist of Rationals.
    // constexpr auto discard2 = make_matrix(make_row(std::integral_constant<int, 3>())):
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 * End of very basic matrix API tests.
 *******************************************************************************/

/// Replace row I (0-based) with given row in the matrix.
template <auto I, class Row, class... Rows>
constexpr auto replace_row(Matrix<Rows...> mat, Row) noexcept
{
    static_assert(I >= 0 && I < mat.count);
    if constexpr (I == 0)
    {
        return Matrix<Row>().append(mat.tail());
    }
    else
    {
        return make_matrix(mat.head()).append(replace_row<I - 1>(mat.tail(), Row()));
    }
}

/// Swap the rows given by indices `I` and `J` in the given matrix.
template <auto I, auto J, class... Rows>
constexpr auto swap_rows(Matrix<Rows...> mat)
{
    if constexpr (I > J)
    {
        return swap_rows<J, I>(Matrix<Rows...>());
    }
    else
    {
        if constexpr (I == 0)
        {
            return make_matrix(get_row<J>(mat)).append(replace_row<J - 1>(mat.tail(), mat.head()));
        }
        else
        {
            return make_matrix(mat.head()).append(swap_rows<I - 1, J - 1>(mat.tail()));
        }
    }
}

/********************************************************************************
 * Test swapping 2 matrix rows; needed for LU pivoting when required.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::MetaLinAlg] Swap matrix rows")
{
    constexpr auto mat = make_matrix(
        make_row(rational<1, 1>, rational<1, 2>, rational<1, 3>),
        make_row(rational<2, 1>, rational<2, 2>, rational<2, 3>),
        make_row(rational<3, 1>, rational<3, 2>, rational<3, 3>));

    constexpr auto swapped = swap_rows<1, 2>(mat);
    REQUIRE(get_row<2>(swapped) == get_row<1>(mat));
    REQUIRE(get_row<1>(swapped) == get_row<2>(mat));
    REQUIRE(get_row<0>(swapped) == get_row<0>(mat));
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 * End test of row swap.
 *******************************************************************************/

/*!
 * @brief Find a row of the matrix with a non-zero on the given diagonal.
 * 
 * @tparam D The index of the column where we want a non-zero.
 * @tparam FIRST The index of the first row where we will look.
 */
template <auto COL, auto FIRST, class... Rows>
constexpr auto find_nonzero_entry(Matrix<Rows...>) noexcept
{
    static_assert(FIRST < sizeof...(Rows), "Non-zero pivot not found!");
    constexpr auto first_row = get_row<FIRST>(Matrix<Rows...>());
    if constexpr (get<COL>(first_row) != Rationals::rational<0>)
    {
        return FIRST;
    }
    else
    {
        return find_nonzero_entry<COL, FIRST + 1>(Matrix<Rows...>());
    }
}

/********************************************************************************
 * Test the ability to find a row with non-zero diagonal element.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::MetaLinAlg] Find non-zero pivot element")
{
    constexpr auto mat = make_matrix(
        make_row(rational<1, 1>, rational<0, 2>, rational<1, 3>),
        make_row(rational<2, 1>, rational<0, 2>, rational<2, 3>),
        make_row(rational<3, 1>, rational<3, 2>, rational<3, 3>));

    REQUIRE(find_nonzero_entry<0, 0>(mat) == 0);
    REQUIRE(find_nonzero_entry<1, 0>(mat) == 2);
    REQUIRE(find_nonzero_entry<0, 1>(mat) == 1);
    REQUIRE(find_nonzero_entry<1, 1>(mat) == 2);
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 *******************************************************************************/

/*!
 * @brief Set the `I`-th element in `row` to the given `Rational` value
 */
template <auto I, auto N, auto D, class... Nums>
constexpr auto set_element(MatrixRow<Nums...> row,
                           Rationals::Rational<N, D>) noexcept
{
    static_assert(I >= 0 && I < sizeof...(Nums));
    if constexpr (I == 0)
    {
        return MatrixRow<Rationals::Rational<N, D>>().append(row.tail());
    }
    else
    {
        return make_row(row.head()).append(set_element<I - 1>(row.tail(), Rationals::Rational<N, D>()));
    }
}

/// Internal recursive implementation of `eliminate_row`.
template <auto I, auto J, int K, class Row, class... Rows>
constexpr auto eliminate_row_helper(Matrix<Rows...>, Row, std::integral_constant<int, K>)
{
    [[maybe_unused]] constexpr auto A = Matrix<Rows...>();
    constexpr auto row = Row();
    [[maybe_unused]] constexpr auto mult = get_elt<J, I>(A) / get_elt<I, I>(A);

    if constexpr (K == I)
    {
        constexpr auto newrow = set_element<I>(row, mult);
        return eliminate_row_helper<I, J>(A, newrow, std::integral_constant<int, K + 1>());
    }
    else if constexpr (K == sizeof...(Rows))
    {
        return row;
    }
    else
    {
        constexpr auto newrow = set_element<K>(row,
                                               get_elt<J, K>(A) - get_elt<I, K>(A) * mult);
        return eliminate_row_helper<I, J>(A, newrow, std::integral_constant<int, K + 1>());
    }
}

/*!
 * @brief Performs the row elimination step in LU factorization.
 *
 * Given template arguments `I`, `J`, eliminates the `I`-th element in row `J`
 * by subtracting a multiple of row `I`.
 */
template <auto I, auto J, class... Rows>
constexpr auto eliminate_row(Matrix<Rows...>)
{
    static_assert(J > I);
    constexpr auto A = Matrix<Rows...>();
    constexpr auto newrow = eliminate_row_helper<I, J>(A, get_row<J>(A),
                                                       std::integral_constant<int, I>());
    return replace_row<J>(A, newrow);
}

/********************************************************************************
 * Test eliminating a row from a matrix (in the LU process). Rather than do this
 * in place I keep a separate L and U around for ease of implementation.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::MetaLinAlg] Do row elimination on L and U factors")
{
    SUBCASE("A simple 2 x 2 matrix")
    {
        constexpr auto A = make_matrix(
            make_row(rational<2>, rational<3>),
            make_row(rational<1>, rational<4>));

        auto LU = eliminate_row<0, 1>(A);
        REQUIRE(LU == Matrix<
                          MatrixRow<Rational<2, 1>, Rational<3, 1>>,
                          MatrixRow<Rational<1, 2>, Rational<5, 2>>>());
    }

    SUBCASE("A 3 x 3 matrix")
    {
        constexpr auto A = make_matrix(
            make_row(rational<2>, -rational<1>, rational<3>),
            make_row(rational<4>, rational<2>, rational<1>),
            make_row(-rational<6>, -rational<1>, rational<2>));

        auto LU = eliminate_row<0, 2>(eliminate_row<0, 1>(A));
        REQUIRE(LU == make_matrix(
                          make_row(rational<2>, -rational<1>, rational<3>),
                          make_row(rational<2>, rational<4>, -rational<5>),
                          make_row(-rational<3>, -rational<4>, rational<11>)));
    }
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 *******************************************************************************/

/// Recursive implementation of the Gaussian elimination step in LU factorization
template <int M, int N, class... Rows>
constexpr auto do_row_elimination(Matrix<Rows...>) noexcept
{
    if constexpr (N == sizeof...(Rows))
    {
        return Matrix<Rows...>();
    }
    else
    {
        constexpr auto LU = eliminate_row<M, N>(Matrix<Rows...>());
        return do_row_elimination<M, N + 1>(LU);
    }
}

/*!
 * @brief LU factorization with row pivoting.
 * 
 * This overload is the recursive implementation; first argument is the current
 * packed `LU` matrix; second is the list of swaps; the last is the row currently
 * to be processed. First entry to factorize `A` should be
 * `factorize(A, typeconst_list<>(), 0)`.
 */
template <class... Rows, int... swaps, int M>
constexpr auto factorize(Matrix<Rows...>,
                         typeconst_list<std::integral_constant<int, swaps>...>,
                         std::integral_constant<int, M>) noexcept
{
    if constexpr (M == sizeof...(Rows) - 1)
    {
        return std::tuple(Matrix<Rows...>(),
                          typeconst_list<std::integral_constant<int, swaps>...>());
    }
    else
    {
        constexpr auto swap = find_nonzero_entry<M, M>(Matrix<Rows...>());
        if constexpr (swap != M)
        {
            constexpr auto LU = swap_rows<M, swap>(Matrix<Rows...>());
            constexpr auto P = typeconst_list<std::integral_constant<int, swaps>...,
                                              std::integral_constant<int, swap>>();
            return factorize(LU, P, std::integral_constant<int, M>());
        }
        else
        {
            constexpr auto LU = do_row_elimination<M, M + 1>(Matrix<Rows...>());
            // If number of swaps is the same as M we didn't do a swap for this
            // row yet; add the index of this row to swaps (indicates no swap).
            if constexpr (sizeof...(swaps) == M)
            {
                constexpr auto P = typeconst_list<std::integral_constant<int, swaps>...,
                                                  std::integral_constant<int, M>>();
                return factorize(LU, P, std::integral_constant<int, M + 1>());
            }
            // otherwise, we did a swap and should not add anything to the swaps
            else
            {
                constexpr auto P = typeconst_list<std::integral_constant<int, swaps>...>();
                return factorize(LU, P, std::integral_constant<int, M + 1>());
            }
        }
    }
}

/*!
 * @brief LU factorization with partial pivoting.
 * 
 * @returns A tuple `[LU, P]` where `LU` is a `Matrix` containing the packed
 * `L` and `U` factors (convention is that diagonal elements of `L` are 1), and
 * `P` allows reversal of the swaps made by pivoting. Apply the inverse of `P`
 * by using `apply_permutation`.
 *
 * @param A Matrix to be factorized.
 */
template <class... Rows>
constexpr auto factorize(Matrix<Rows...>) noexcept
{
    constexpr auto P = typeconst_list<>();
    return factorize(Matrix<Rows...>(), P, std::integral_constant<int, 0>());
}

/********************************************************************************
 * Test LU factorization with pivoting to fix zero diagonals.
 *******************************************************************************/
#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::MetaLinAlg] Test LU factorization")
{
    SUBCASE("Test a matrix that doesn't require pivoting")
    {
        constexpr auto A = make_matrix(
            make_row(rational<2>, rational<-1>, rational<3>),
            make_row(rational<4>, rational<2>, rational<1>),
            make_row(rational<-6>, rational<-1>, rational<2>));

        auto [LU, P] = factorize(A);

        REQUIRE(LU == make_matrix(
                          make_row(rational<2>, -rational<1>, rational<3>),
                          make_row(rational<2>, rational<4>, -rational<5>),
                          make_row(-rational<3>, -rational<1>, rational<6>)));

        REQUIRE(P == make_list(
                         std::integral_constant<int, 0>(),
                         std::integral_constant<int, 1>()));
    }

    SUBCASE("A more difficult case with pivoting")
    {
        constexpr auto A = make_matrix(
            make_row(rational<1>, rational<2>, rational<1>, rational<0>),
            make_row(rational<0>, rational<0>, rational<3>, rational<1>),
            make_row(rational<5>, rational<0>, rational<2>, rational<3>),
            make_row(rational<1>, rational<1>, rational<1>, rational<1>));

        auto [LU, P] = factorize(A);

        REQUIRE(LU == make_matrix(
                          make_row(rational<1>, rational<2>, rational<1>, rational<0>),
                          make_row(rational<5>, -rational<10>, -rational<3>, rational<3>),
                          make_row(rational<0>, rational<0>, rational<3>, rational<1>),
                          make_row(rational<1>, rational<1, 10>, rational<1, 10>, rational<3, 5>)));

        REQUIRE(P == make_list(
                         std::integral_constant<int, 0>(),
                         std::integral_constant<int, 2>(),
                         std::integral_constant<int, 2>()));
    }
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 *******************************************************************************/

/// Recursive implementation of `apply_permutation`.
template <auto M, int... swaps, class Row>
constexpr auto apply_swaps(typeconst_list<std::integral_constant<int, swaps>...>,
                           Row)
{
    [[maybe_unused]] constexpr auto lst = typeconst_list<std::integral_constant<int, swaps>...>();
    if constexpr (M == sizeof...(swaps))
    {
        return Row();
    }
    else
    {
        constexpr auto swap = get<M>(lst)();
        if constexpr (swap != M)
        {
            // swap_rows was defined on a `Matrix` but `Matrix` is just a typedef of
            // typeconst_list so it will work just as well here.
            constexpr auto row = swap_rows<M, swap>(Row());
            return apply_swaps<M + 1>(lst, row);
        }
        else
        {
            return apply_swaps<M + 1>(lst, Row());
        }
    }
}

/// Helper function for `backsub_fwd`.
template <auto start, class Arow, class Soln, class Entry>
constexpr auto subtract_previous(Arow, Soln, Entry)
{
    static_assert(start <= Soln::count);
    if constexpr (start == Soln::count)
    {
        return Entry();
    }
    else
    {
        constexpr auto increment = get<start>(Soln()) * get<start>(Arow());
        return subtract_previous<start + 1>(Arow(), Soln(), Entry() - increment);
    }
}

/// Helper function for `backsub_back`.
template <auto start, auto sz, class Arow, class Soln, class Entry>
constexpr auto subtract_latter(Arow, Soln, Entry)
{
    static_assert(start <= Arow::count);
    if constexpr (start == Arow::count)
    {
        return Entry();
    }
    else
    {
        constexpr auto increment = get<start - sz + Soln::count>(Soln()) * get<start>(Arow());
        return subtract_latter<start+1, sz>(Arow(), Soln(), Entry() - increment);
    }
}

/// Recursive implementation of `backsub_fwd`.
template <auto M, class... Rows, class Rhs, class Soln>
constexpr auto backsub_fwd_impl(Matrix<Rows...>, Rhs, Soln)
{
    if constexpr (M == sizeof...(Rows))
    {
        return Soln();
    }
    else if constexpr (M == 0)
    {
        static_assert(Soln::count == 0);
        return backsub_fwd_impl<1>(Matrix<Rows...>(), Rhs(), make_row(get<0>(Rhs())));
    }
    else
    {
        constexpr auto entry = get<M>(Rhs());
        constexpr auto soln = Soln::append(
            make_row(subtract_previous<0>(get_row<M>(Matrix<Rows...>()), Soln(), entry)));
        return backsub_fwd_impl<M + 1>(Matrix<Rows...>(), Rhs(), soln);
    }
}

/*!
 * @brief Perform forward substitution for lower triangular matrix
 * 
 * The diagonal of the matrix is taken to be 1. `backsub_fwd(A, b)` returns a
 * vector `x` (in the form of a `typeconst_list`) satisying `A*x = b`.
 * 
 * @param A Matrix specifying the linear operator
 * @param b The right-hand side of the system
 * 
 * @returns `x` solving `A * x = b`.
 */
template <class... Rows, class Rhs>
constexpr auto backsub_fwd(Matrix<Rows...>, Rhs)
{
    return backsub_fwd_impl<0>(Matrix<Rows...>(), Rhs(), MatrixRow<>());
}

/// Recursive implementation of backsubstitution.
template <int M, class... Rows, class Rhs, class Soln>
constexpr auto backsub_back_impl(Matrix<Rows...>, Rhs, Soln)
{
    static_assert(M >= -1 && M < static_cast<long int>(sizeof...(Rows)));
    if constexpr (M == -1)
    {
        return Soln();
    }
    else if constexpr (M == sizeof...(Rows)-1)
    {
        static_assert(Soln::count == 0);
        return backsub_back_impl<M-1>(
            Matrix<Rows...>(), Rhs(),
            make_row(get<M>(Rhs()) / get_elt<M, M>(Matrix<Rows...>()))
        );
    }
    else
    {
        constexpr auto entry = get<M>(Rhs());
        constexpr auto soln = make_row(
            subtract_latter<M+1, sizeof...(Rows)>(get_row<M>(Matrix<Rows...>()), Soln(), entry)
            / get_elt<M, M>(Matrix<Rows...>())
        ).append(Soln());
        return backsub_back_impl<M - 1>(Matrix<Rows...>(), Rhs(), soln);
    }
}

/*!
 * @brief Perform back substitution on an upper triangular matrix.
 */
template <class... Rows, class Rhs>
constexpr auto backsub_back(Matrix<Rows...>, Rhs)
{
    constexpr auto M = sizeof...(Rows);
    return backsub_back_impl<M-1>(Matrix<Rows...>(), Rhs(), MatrixRow<>());
}

/*!
 * @brief Use back-substitution to solve `Ax = b`, where `A` is a packed `LU`
 * factorization.
 */
template <class... Rows, class Rhs>
constexpr auto backsub(Matrix<Rows...>, Rhs)
{
    constexpr auto y = backsub_fwd(Matrix<Rows...>(), Rhs());
    return backsub_back(Matrix<Rows...>(), y);
}

/*!
 * @brief Apply the permutation returned by LU factorization to a vector.
 */
template <int... swaps, class Row>
constexpr auto apply_permutation(typeconst_list<std::integral_constant<int, swaps>...>,
                                 Row)
{
    static_assert(sizeof...(swaps) == Row::count - 1);
    return apply_swaps<0>(
        typeconst_list<std::integral_constant<int, swaps>...>(), Row());
}

/*!
 * @brief Solve the system `Ax = b`.
 * 
 * `linear_solve(A, b)` returns `x` satisfying `A*x = b`. This is accomplished
 * by:
 * 
 * - Factoring A into the product `P * L * U` by a standard LU factorization w/
 * partial pivoting
 * - Apply the permutation to the given right-hand side.
 * - Do back-substitution to solve the system.
 * 
 * All computations are performed at compile time.
 * 
 * @param A The matrix giving the linear operator
 * @param b The right-hand side of the system.
 * 
 * @returns `x` such that `A*x = b`.
 */
template <class... Rows, class Row>
constexpr auto linear_solve(Matrix<Rows...>, Row)
{
    constexpr auto factorization = factorize(Matrix<Rows...>());
    constexpr auto LU = get<0>(factorization);
    constexpr auto P = get<1>(factorization);

    constexpr auto rhs = apply_permutation(P, Row());
    return backsub(LU, rhs);
}

/********************************************************************************
 * Test a linear equation solve using LU factorization.
 *******************************************************************************/

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[Galerkin::MetaLinAlg] Test full linear solve")
{
    constexpr auto A = make_matrix(
        make_row(rational<1>, rational<2>, rational<1>, rational<0>),
        make_row(rational<0>, rational<0>, rational<3>, rational<1>),
        make_row(rational<5>, rational<0>, rational<2>, rational<3>),
        make_row(rational<1>, rational<1>, rational<1>, rational<1>));
    constexpr auto b = make_row(
        rational<1>, rational<2>, rational<3>, rational<4>);

    constexpr auto x = linear_solve(A, b);

    REQUIRE(x == make_row(
                     -rational<2>, rational<2>, -rational<1>, rational<5>));
}

#endif /* DOCTEST_LIBRARY_INCLUDED */

/********************************************************************************
 *******************************************************************************/

} // namespace MetaLinAlg

} // namespace Galerkin

#endif /* METALINALG_HPP */
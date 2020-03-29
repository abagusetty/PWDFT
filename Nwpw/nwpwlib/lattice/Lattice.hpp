#ifndef _LATTICE_HPP_
#define _LATTICE_HPP_
/* Lattice.hpp
   Author - Eric Bylaska

*/


class Lattice {

   double punita[9],punitg[9],pecut,pwcut,pomega;

public:

   /* constructor */
   Lattice();

   double unita(const int i, const int j) { return punita[i+j*3]; }
   double unitg(const int i, const int j) { return punitg[i+j*3]; }
   double ecut() { return pecut; }
   double wcut() { return pwcut; }
   double omega() { return pomega; }
   double eggcut() { return 2*pecut; }
   double wggcut() { return 2*pwcut; }
};

#endif
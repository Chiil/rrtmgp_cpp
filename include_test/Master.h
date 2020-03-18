/*
 * MicroHH
 * Copyright (c) 2011-2017 Chiel van Heerwaarden
 * Copyright (c) 2011-2017 Thijs Heus
 * Copyright (c) 2014-2017 Bart van Stratum
 *
 * This file is part of MicroHH
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MASTER_H
#define MASTER_H

#include <cstdarg>
#include <sstream>
#include <string>

struct MPI_data
{
    int nprocs;
    int npx;
    int npy;
    int mpiid;
    int mpicoordx;
    int mpicoordy;
};

class Master
{
    public:
        Master();
        ~Master();

        void start();
        void init();

        // Overload the broadcast function.
        void broadcast(char*, int, int=0);
        void broadcast(int*, int, int=0);
        void broadcast(bool*, int, int=0);
        void broadcast(double*, int, int=0);
        void broadcast(float*, int, int=0);
        void broadcast(unsigned long*, int, int=0);

        // Overload the sum function.
        void sum(int*, int);
        void sum(double*, int);
        void sum(float*, int);

        // Overload the max function.
        void max(double*, int);
        void max(float*, int);

        // Overload the min function.
        void min(double*, int);
        void min(float*, int);

        void print_message(const char *format, ...);
        void print_message(const std::ostringstream&);
        void print_message(const std::string&);

        void print_warning(const char *format, ...);
        void print_warning(const std::ostringstream&);
        void print_warning(const std::string&);

        void print_error  (const char *format, ...);

        int get_mpiid() const { return md.mpiid; }
        const MPI_data& get_MPI_data() const { return md; }

    private:
        bool initialized;
        bool allocated;

        MPI_data md;
};

// Implementation below
inline Master::Master()
{
    initialized = false;
    allocated   = false;
}

inline Master::~Master()
{
    print_message("Finished run on %d processes\n", md.nprocs);
}

inline void Master::start()
{
    initialized = true;

    // Set the rank of the only process to 0.
    md.mpiid = 0;
    // Set the number of processes to 1.
    md.nprocs = 1;

    print_message("Starting run on %d processes\n", md.nprocs);
}

inline void Master::init()
{
    md.npx = 1;
    md.npy = 1;

    // set the coordinates to 0
    md.mpicoordx = 0;
    md.mpicoordy = 0;

    allocated = true;
}

// All broadcasts return directly, because there is nothing to broadcast.
inline void Master::broadcast(char* data, int datasize, int mpiid_to_write) {}
inline void Master::broadcast(int* data, int datasize, int mpiid_to_write) {}
inline void Master::broadcast(unsigned long* data, int datasize, int mpiid_to_write) {}
inline void Master::broadcast(double* data, int datasize, int mpiid_to_write) {}
inline void Master::broadcast(float* data, int datasize, int mpiid_to_write) {}
inline void Master::sum(int* var, int datasize) {}
inline void Master::sum(double* var, int datasize) {}
inline void Master::sum(float* var, int datasize) {}
inline void Master::max(double* var, int datasize) {}
inline void Master::max(float* var, int datasize) {}
inline void Master::min(double* var, int datasize) {}
inline void Master::min(float* var, int datasize) {}

inline void Master::print_message(const char *format, ...)
{
    if (md.mpiid == 0)
    {
        va_list args;
        va_start(args, format);
        std::vfprintf(stdout, format, args);
        va_end(args);
    }
}

inline void Master::print_message(const std::ostringstream& ss)
{
    if (md.mpiid == 0)
        std::cout << ss.str();
}

inline void Master::print_message(const std::string& s)
{
    if (md.mpiid == 0)
        std::cout << s << std::endl;
}

inline void Master::print_warning(const char *format, ...)
{
    std::string warningstr("WARNING: ");
    warningstr += std::string(format);

    const char *warningformat = warningstr.c_str();

    if (md.mpiid == 0)
    {
        va_list args;
        va_start(args, format);
        std::vfprintf(stdout, warningformat, args);
        va_end(args);
    }
}

inline void Master::print_warning(const std::ostringstream& ss)
{
    if (md.mpiid == 0)
        std::cout << "WARNING: " << ss.str();
}

inline void Master::print_warning(const std::string& s)
{
    if (md.mpiid == 0)
        std::cout << "WARNING: " << s << std::endl;
}

inline void Master::print_error(const char *format, ...)
{
    std::string errorstr("ERROR: ");
    errorstr += std::string(format);

    const char *errorformat = errorstr.c_str();

    if (md.mpiid == 0)
    {
        va_list args;
        va_start(args, format);
        std::vfprintf(stdout, errorformat, args);
        va_end(args);
    }
}
#endif

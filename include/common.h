
/*
 * File:   common.h
 * Author: Emilio Vicari, Michele Amoretti
 *
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>

// parallel architecture constants

// defines for bitwise operations
#define register_t unsigned int
#define BYTES_PER_REG sizeof(register_t)
#define BITS_PER_REG (BYTES_PER_REG * 8)
#define SELECTION_BITMASK ((register_t)1)
#define MAX_SYSTEM_SIZE BITS_PER_REG

static void handleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (handleError( err, __FILE__, __LINE__ ))
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


#endif /* COMMON_H */

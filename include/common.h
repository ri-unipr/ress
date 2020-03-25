/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   common.h
 * Author: e.vicari
 *
 * Created on 2 marzo 2016, 10.52
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

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


#endif /* COMMON_H */



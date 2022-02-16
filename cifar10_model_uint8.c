
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


/* Autotiler includes. */
#include "cifar10_model_uint8.h"
#include "cifar10_model_uint8Kernels.h"
#include "gaplib/ImgIO.h"
#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s 
#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

AT_HYPERFLASH_FS_EXT_ADDR_TYPE cifar10_model_uint8_L3_Flash = 0;

/* Inputs */
char *ImageName;
#ifdef MODEL_NE16
L2_MEM unsigned char Input_1[3072];
#else
L2_MEM signed char Input_1[3072];
#endif
/* Outputs */
L2_MEM short int Output_1[10];

static void cluster()
{
    #ifdef PERF
    printf("Start timer\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

    cifar10_model_uint8CNN(Input_1, Output_1);
    printf("Runner completed\n");
    int MaxPred=Output_1[0], PredClass=0;
    for (int i=1; i<10; i++){
        if (Output_1[i] > MaxPred) {
            MaxPred = Output_1[i];
            PredClass = i;
        }
    }
    printf("Predicted Class: %d with confidence: %d\n", PredClass, MaxPred);

}

int test_cifar10_model_uint8(void)
{
    printf("Entering main controller\n");
    /* ----------------> 
     * Put here Your input settings
     * <---------------
     */

    ImageName = __XSTR(AT_IMAGE);
    #if defined(MODEL_HWC) || defined(MODEL_NE16)
    int Traspose2CHW = 0;
    #else
    int Traspose2CHW = 1;
    #endif
    printf("Reading image in %s\n", Traspose2CHW?"CHW":"HWC");
    if (ReadImageFromFile(ImageName, 32, 32, 3, Input_1, 32*32*3*sizeof(char), IMGIO_OUTPUT_CHAR, Traspose2CHW)) {
        printf("Failed to load image %s\n", ImageName);
        pmsis_exit(-1);
    }
    // NE16 takes unsigned input with zero point implied (-128)
    #if !defined(MODEL_NE16)
    for (int i=0; i<32*32*3; i++) Input_1[i] -= 128;
    #endif

#ifndef __EMUL__
    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    cl_conf.id = 0;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    if (cur_fc_freq == -1)
    {
        printf("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-4);
    }

    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    if (cur_cl_freq == -1)
    {
        printf("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-5);
    }
#ifdef __GAP9__
    pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_FC*1000*1000);
#endif
#endif
    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("Constructor\n");
    int ConstructorErr = cifar10_model_uint8CNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: %d\n(check the generated file cifar10_model_uint8Kernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        pmsis_exit(-6);
    }


    printf("Call cluster\n");
#ifndef __EMUL__
    struct pi_cluster_task task = {0};
    task.entry = cluster;
    task.arg = NULL;
    task.stack_size = (unsigned int) STACK_SIZE;
    task.slave_stack_size = (unsigned int) SLAVE_STACK_SIZE;

    pi_cluster_send_task_to_cl(&cluster_dev, &task);
#else
    cluster();
#endif

    cifar10_model_uint8CNN_Destruct();

#ifdef PERF
    {
      unsigned int TotalCycles = 0, TotalOper = 0;
      printf("\n");
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        printf("%45s: Cycles: %10u, Operations: %10u, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], AT_GraphOperInfosNames[i], ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
        TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
      }
      printf("\n");
      printf("%45s: Cycles: %10u, Operations: %10u, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
      printf("\n");
    }
#endif

    printf("Ended\n");
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** NNTOOL cifar10_model_uint8 Example ***\n\n");
    #ifdef __EMUL__
    test_cifar10_model_uint8();
    #else
    return pmsis_kickoff((void *) test_cifar10_model_uint8);
    #endif
    return 0;
}

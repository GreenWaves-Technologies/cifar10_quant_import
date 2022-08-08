
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


/* Autotiler includes. */
#include "cifar10_model.h"
#include "cifar10_modelKernels.h"
#include "cifar10_modelInfo.h"
#include "gaplib/ImgIO.h"
#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s 
#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

AT_HYPERFLASH_FS_EXT_ADDR_TYPE cifar10_model_L3_Flash = 0;

static char * ClassDict[10] = {"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
/* Inputs */
char *ImageName;
#ifdef MODEL_NE16
L2_MEM unsigned char Input_1[3072];
#else
L2_MEM signed char Input_1[3072];
#endif
/* Outputs */
#ifdef OUTPUT_SHORT
L2_MEM short int Output_1[10];
#else
#ifdef MODEL_NE16
L2_MEM unsigned char Output_1[10];
#else
L2_MEM signed char Output_1[10];
#endif
#endif

int MaxPred, PredClass;
static void cluster()
{
    #ifdef PERF
    printf("Start timer\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

    cifar10_modelCNN(Input_1, Output_1);
    printf("Runner completed\n");
    MaxPred=Output_1[0], PredClass=0;
    for (int i=1; i<10; i++){
        printf("Class: %10s --> %5.2f\n", ClassDict[i], (((float) Output_1[i]) - cifar10_model_Output_1_OUT_ZERO_POINT) * cifar10_model_Output_1_OUT_SCALE);
        if (Output_1[i] > MaxPred) {
            MaxPred = Output_1[i];
            PredClass = i;
        }
    }
    printf("Predicted Class: %10s with confidence: %.2f\n", ClassDict[PredClass], (((float) MaxPred) - cifar10_model_Output_1_OUT_ZERO_POINT) * cifar10_model_Output_1_OUT_SCALE);

}

int test_cifar10_model(void)
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
    pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;
    cl_conf.cc_stack_size = STACK_SIZE;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }
    pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_FC*1000*1000);
    printf("Set FC Frequency = %d MHz, CL Frequency = %d MHz, PERIIPH Frequency = %d MHz\n",
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));
    #ifdef VOLTAGE
    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
    printf("Voltage: %dmV\n", VOLTAGE);
    #endif
#endif
    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("Constructor\n");
    int ConstructorErr = cifar10_modelCNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: %d\n(check the generated file cifar10_modelKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        pmsis_exit(-6);
    }


    printf("Call cluster\n");
#ifndef __EMUL__
    struct pi_cluster_task task;
    pi_cluster_task(&task, cluster, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);
    pi_cluster_send_task_to_cl(&cluster_dev, &task);
#else
    cluster();
#endif

    cifar10_modelCNN_Destruct();

#ifdef PERF
    {
      unsigned int TotalCycles = 0, TotalOper = 0;
      printf("\n");
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
      }
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
      }
      printf("\n");
      printf("%45s: Cycles: %12u, Cyc%%: 100.0%%, Operations: %12u, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
      printf("\n");
    }
#endif
    #ifdef CI
    if (PredClass == CI) {printf("Correct Results\n"); pmsis_exit(0);}
    else {printf("Wrong Results\n"); pmsis_exit(-1);}
    #endif

    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** NNTOOL cifar10_model Example ***\n\n");
    #ifdef __EMUL__
    test_cifar10_model();
    #else
    return pmsis_kickoff((void *) test_cifar10_model);
    #endif
    return 0;
}

/********** ********** ********** ********** ********** ********** **********
 *                                 ( 0-0 )                                  *
 *                            (" \/       \/ ")                             *
 ********** ********** ********** ********** ********** ********** **********
 * Copyright (C) 2018 - 2022 CURI & HKCLR                                   *
 * File name   : curi_udp.h                                                 *
 * Author      : CHEN Wei                                                   *
 * Version     : 1.0.0                                                      *
 * Date        : 2022-04-29                                                 *
 * Description : Udp communication.                                         *
 * Others      : None                                                       *
 * History     : 2022-04-29 1st version.                                    *
 ********** ********** ********** ********** ********** ********** **********
 *                              (            )                              *
 *                               \ __ /\ __ /                               *
 ********** ********** ********** ********** ********** ********** **********/

#ifndef CURI_UDP_H
#define CURI_UDP_H

#ifdef WIN32
	#include <Winsock2.h>
	#pragma comment(lib, "WS2_32.lib")
#else
	#include <arpa/inet.h> 
	#include <netinet/in.h> 
	#include <sys/socket.h> 
	#include <sys/types.h> 
	#include <stdint.h>
#endif
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 

#define MAX_REMOTER_DATA_SIZE 2048

typedef struct _udp_node
{
	int CURI_SEND_PORT;
	int CURI_RECIVE_PORT;
	char recieve_buffer[MAX_REMOTER_DATA_SIZE];
	char send_buffer[MAX_REMOTER_DATA_SIZE];
	int recieve_size;
	int send_size;
	int server_fd;
	int client_fd;
	struct sockaddr_in server_addr;
	struct sockaddr_in client_addr;
	fd_set rset;
    uint8_t lock;
}udp_node;

#ifdef __cplusplus
extern "C" 
{
#endif

// #define lock_udp_node(udp_node_p, lock_state) \
//     if (lock_state) { \ 
// 		int udp_lock_cnts = 0; \
// 		while (udp_node_p->lock && udp_lock_cnts < 50) { udp_lock_cnts ++; sleep_period(10); } \
// 		if (udp_lock_cnts >= 50) return; \
// 	} udp_node_p->lock = lock_state; 

/*
 * function: udp_init
 *     udp communication initialization
 * input:
 *     p[udp_node *]: the udp_node structure
 *     local_ip[char *]: the local node ip
 *     send_port[int]: data send port
 *     recieve_port[int]: recieve command port
 * output:
 *     state[int]: success return 0
 */
int udp_init(udp_node* p, char local_ip[], char remote_ip[], int send_port, int recieve_port);

/*
 * function: udp_select
 *     udp communication selection to test if there are some data at the port
 * input:
 *     p[udp_node *]: the udp_node structure
 *     usec[int]: wait data time in us
 * output:
 *     [int]: the data recieved
 */
int udp_select(udp_node* p, int usec);

/*
 * function: udp_send
 *     udp communication send data to the remote port
 * input:
 *     p[udp_node *]: the udp_node structure
 * output:
 *     [void]
 */
void udp_send(udp_node* p);

/*
 * function: udp_receive
 *     udp communication receive data from the local port
 * input:
 *     p[udp_node *]: the udp_node structure
 * output:
 *     [void]
 */
void udp_receive(udp_node* p);

/*
 * function: udp_print
 *     udp communication print the recieve data from the local port
 * input:
 *     p[udp_node *]: the udp_node structure
 * output:
 *     [void]
 */
void udp_print(udp_node* p);

/*
 * function: udp_close
 *     udp communication close the port
 * input:
 *     p[udp_node *]: the udp_node structure
 * output:
 *     [void]
 */
void udp_close(udp_node* p);

#ifdef __cplusplus
}
#endif
#endif
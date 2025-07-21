/********** ********** ********** ********** ********** ********** **********
 *                                 ( 0-0 )                                  *
 *                            (" \/       \/ ")                             *
 ********** ********** ********** ********** ********** ********** **********
 * Copyright (C) 2018 - 2022 CURI & HKCLR                                   *
 * File name   : curi_udp.c                                                 *
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

#include "curi_udp.h"

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
int udp_init(udp_node* p, char local_ip[], char remote_ip[], int send_port, int recieve_port)
{
#ifdef WIN32
	WSADATA wsaData;
	int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (err != 0) {
		return err;
	}

	if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
		WSACleanup();
		return -1;
	}
#endif 
    p->lock = 0;
	// create UDP socket 
	p->server_fd = socket(AF_INET, SOCK_DGRAM, 0);
	p->server_addr.sin_addr.s_addr = inet_addr(local_ip);
	p->server_addr.sin_port = htons(send_port);
	p->server_addr.sin_family = AF_INET;

	p->client_fd = socket(AF_INET, SOCK_DGRAM, 0);
	p->client_addr.sin_addr.s_addr = inet_addr(remote_ip);
	p->client_addr.sin_port = htons(recieve_port);
	p->client_addr.sin_family = AF_INET;

	// bind server address to socket descriptor 
#ifdef WIN32
	if (bind(p->server_fd, (SOCKADDR*)&(p->server_addr)), sizeof(p->server_addr)) == -1) {
#else
	if (bind(p->server_fd, (struct sockaddr_in *)&(p->server_addr), sizeof(p->server_addr)) == -1) {
#endif
		perror("bind error.");
		return -1;
	}

	// set the buffer size
	int buffer_size = MAX_REMOTER_DATA_SIZE;
	if (0 != setsockopt(p->server_fd, SOL_SOCKET, SO_SNDBUF, (const char*)&buffer_size, sizeof(int)))
		return -2;
	else if (0 != setsockopt(p->client_fd, SOL_SOCKET, SO_RCVBUF, (const char*)&buffer_size, sizeof(int)))
		return -3;
	//else if (0 != setsockopt(p->client_fd, SOL_SOCKET, SO_REUSEADDR, 0, sizeof(int)))
	//	return -4;

	FD_ZERO(&(p->rset));

	return 0;
}

/*
 * function: udp_select
 *     udp communication selection to test if there are some data at the port
 * input:
 *     p[udp_node *]: the udp_node structure
 *     usec[int]: wait data time in us
 * output:
 *     [int]: the data recieved
 */
int udp_select(udp_node* p, int usec)
{
	FD_SET(p->server_fd, &(p->rset));
	// select the ready descriptor 
	struct timeval t;
	t.tv_sec = 0;
	t.tv_usec = usec;
	int nready = select(p->server_fd + 1, &(p->rset), NULL, NULL, &t);
	if (FD_ISSET(p->server_fd, &(p->rset))) {
		memset(p->recieve_buffer, 0, sizeof(p->recieve_buffer));
		p->recieve_size = recv(p->server_fd, p->recieve_buffer, MAX_REMOTER_DATA_SIZE, 0);
	} else {
		p->recieve_size = 0;
	}
	return p->recieve_size;
}

/*
 * function: udp_send
 *     udp communication send data to the remote port
 * input:
 *     p[udp_node *]: the udp_node structure
 * output:
 *     [void]
 */
void udp_send(udp_node* p)
{
	// lock_udp_node(p, 1);
	char send_buffer[MAX_REMOTER_DATA_SIZE];
	strncpy(send_buffer, p->send_buffer, strlen(p->send_buffer));
	sendto(p->client_fd, send_buffer, strlen(send_buffer), 0,
		(const struct sockaddr_in*)&(p->client_addr), sizeof(p->client_addr));
	// lock_udp_node(p, 0);
}

/*
 * function: udp_receive
 *     udp communication receive data from the local port
 * input:
 *     p[udp_node *]: the udp_node structure
 * output:
 *     [void]
 */
void udp_receive(udp_node* p)
{
	p->recieve_size = recv(p->server_fd, p->recieve_buffer, MAX_REMOTER_DATA_SIZE, 0);
}

/*
 * function: udp_print
 *     udp communication print the recieve data from the local port
 * input:
 *     p[udp_node *]: the udp_node structure
 * output:
 *     [void]
 */
void udp_print(udp_node* p)
{
	p->recieve_buffer[p->recieve_size] = 0;
	puts(p->recieve_buffer);
}

/*
 * function: udp_close
 *     udp communication close the port
 * input:
 *     p[udp_node *]: the udp_node structure
 * output:
 *     [void]
 */
void udp_close(udp_node* p)
{
#ifdef WIN32
	closesocket(p->server_fd);
	closesocket(p->client_fd);
	WSACleanup();
#else
	close(p->server_fd);
	close(p->client_fd);
#endif
}

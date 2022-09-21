#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define SHIFT_ROW 0
#define SHIFT_COL 1
#define DISP 1

#define TRUE 1
#define FALSE 0

#define MOVING_AVERAGE_WINDOW 5
#define MAX_NBR_COUNT 4

#define NBR_READING_REQUEST 1
#define NBR_READING_RESPONSE 2
#define SEND_TO_BS 3
#define BS_TAG 4

#define MIN 5700
#define MAX 6300
#define SATELLITE_MIN 6001
#define SATELLITE_MAX 6500
#define THRESHOLD 6000
#define TOLERANCE_RANGE 100

#define MAX_ITERATION 20
#define RESOLVE 10

// Declaration of functions
void *satellite_altimeter();
float generate_height(float min, float max);
float calculate_moving_average(float height_arr[]);
void update_height_arr(float height_arr[], float new_height);

// Structure for each entry
// In the satellite altimeter shared global array
struct satellite_entry
{
    time_t entry_time_recorded;
    float entry_reading;
    int entry_coordinates[2];
};

// Structure for report sent by a node to the base station
struct report 
{
    long reported_time;
    double comm_time;
    float current_reading;
    int nbr_ranks[MAX_NBR_COUNT];
    float nbr_readings[MAX_NBR_COUNT];
    int coordinates[2];
    int nbr_matches;
};

// Satellite altimeter shared global array
struct satellite_entry *satellite;

// Grid dimensions (row x column)
int row;
int col;

// Satellite altimeter terminal signal
int running = TRUE;

int main(int argc, char *argv[]) 
{
    // Initialise MPI environment
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Wtime();

    // Seed the random number generator
    srand(time(NULL) + rank);

    // Process input for grid dimensions
    row = atoi(argv[1]);
    col = atoi(argv[2]);

    // Obtain rank of base station
    int bs_rank = row * col;

    /*
        Create Cartesian topology for nodes
    */

    // Specify grid dimensions
    int cart_dim[2] = {row, col};

    // Periodic shift is false
    int periodic_shift[2] = {0, 0}; 

    // Does not matter if it is true (1) or false (0)
    int reorder = FALSE; 

    // Create new communicator
    MPI_Comm cart_comm;
    int ierr = 0;
    ierr = MPI_Cart_create(MPI_COMM_WORLD, 2, cart_dim, periodic_shift, reorder, &cart_comm);
    if (ierr != 0) printf("Error[%d] Creating CART\n", ierr);

    /*
        Variables
    */

    // POSIX Thread ID
    pthread_t pt_id;

    // Own node's coordinates
    int coordinates[2];

    // Ranks of neighbour nodes
    // Elements in order: Top, bottom, left, right
    int nbr_ranks[MAX_NBR_COUNT] = {-1, -1, -1, -1};

    // Number of neighbour nodes
    int nbr_count = 0;

    // Termination message sent by the base station
    int termination_message = FALSE;

    /* 
        Some base station log file contents
    */
    
    // Number of messages passed to neighbour nodes by a node
    int messages_passed = 0;

    // Total number of messages passed to neighbour nodes by all nodes
    int total_messages_passed = 0;

    // Total messages sent between the node and the base station
    int node_bs_message_count = 2;

    /*
        Create node's report
    */

    MPI_Datatype report_type;
    int lengths[7] = {1, 1, 1, 4, 4, 2, 1};

    MPI_Aint displacements[7];
    struct report report;

    MPI_Datatype types[7] = {MPI_LONG, MPI_DOUBLE, MPI_FLOAT, MPI_INT, MPI_FLOAT, MPI_INT, MPI_INT};

    // Displacement for blocks
    MPI_Aint base_address;
    MPI_Get_address(&report, &base_address);
    MPI_Get_address(&report.reported_time, &displacements[0]);
    MPI_Get_address(&report.comm_time, &displacements[1]);
    MPI_Get_address(&report.current_reading, &displacements[2]);
    MPI_Get_address(&report.nbr_ranks[0], &displacements[3]);
    MPI_Get_address(&report.nbr_readings[0], &displacements[4]);
    MPI_Get_address(&report.coordinates[0], &displacements[5]);
    MPI_Get_address(&report.nbr_matches, &displacements[6]);

    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);
    displacements[3] = MPI_Aint_diff(displacements[3], base_address);
    displacements[4] = MPI_Aint_diff(displacements[4], base_address);
    displacements[5] = MPI_Aint_diff(displacements[5], base_address);
    displacements[6] = MPI_Aint_diff(displacements[6], base_address);

    MPI_Type_create_struct(7, lengths, displacements, types, &report_type);
    MPI_Type_commit(&report_type);

    /*
        Obtain information of own node and neighbour nodes
    */

    if (rank != bs_rank)
    {
        // Obtain and store own node's coordinates
        MPI_Cart_coords(cart_comm, rank, 2, coordinates);

        // Obtain and store top and bottom neighbour nodes' ranks
        MPI_Cart_shift(cart_comm, SHIFT_ROW, DISP, &nbr_ranks[0], &nbr_ranks[1]);

        // Obtain and store left and right neighbour nodes' ranks
        MPI_Cart_shift(cart_comm, SHIFT_COL, DISP, &nbr_ranks[2], &nbr_ranks[3]);

        // Obtain and store number of neighbour nodes
        for (int i = 0; i < MAX_NBR_COUNT; i++)
        {
            if (nbr_ranks[i] >= 0)
            {
                nbr_count++;
            }
        }
    }

    /*
        Create a POSIX thread as the satellite altimeter
    */

    if (rank == bs_rank)
    {
        pthread_create(&pt_id, NULL, satellite_altimeter, NULL);
    }

    /*
        Code for nodes starts here
    */

    if (rank != bs_rank) 
    {
        // Current reading (moving average)
        // Reading is obtained by calling calculate_moving_average 
        float current_reading;

        // Store generated sea water column height values
        float height_arr[MOVING_AVERAGE_WINDOW];
        
        // Store readings of neighbour nodes
        float nbr_readings[MAX_NBR_COUNT] = {0, 0, 0, 0};

        // If neighbour nodes' readings are required, this will be set to true (1)
        int send_reading_request = FALSE;

        // Store either true (1) or false (0)
        // If it is true, it means that there are requests from neighbour nodes
        int reading_request;

        // Number of matches between own node reading and neighbour nodes' readings
        int nbr_matches = 0;

        // Store ranks of neighbour nodes that are valid (neighbour nodes that exist)
        int *valid_nbr_ranks;
        valid_nbr_ranks = (int *) malloc(nbr_count * sizeof(int));

        // Store number of requests received, sent from neighbour nodes
        int nbr_requests_count;

        // Store neighbour nodes' requests
        MPI_Request *nbr_requests;
        nbr_requests = (MPI_Request *) malloc(nbr_count * sizeof(MPI_Request));

        // Store neighbour nodes' responses
        MPI_Request *nbr_responses;
        nbr_responses = (MPI_Request *) malloc(nbr_count * sizeof(MPI_Request));

        // Store indices of the request handlers that has completed the routines
        int *completed_nbr_requests;
        completed_nbr_requests = (int *) malloc(nbr_count * sizeof(int));

        /*
            Be prepared to receive requests from neighbour nodes
        */

        // For loop to receive reading requests from neighbour nodes
        // Then, store the ranks of neighbour nodes that are valid
        int alloc = 0;
        for (int i = 0; i < MAX_NBR_COUNT; i++) 
        {
            if (nbr_ranks[i] >= 0) 
            {
                MPI_Irecv(&reading_request, 1, MPI_INT, nbr_ranks[i], NBR_READING_REQUEST, cart_comm, (nbr_requests + alloc));
                valid_nbr_ranks[alloc] = nbr_ranks[i];
                alloc++;
            }
        }
        alloc = 0;
        
        // Initially, generate 5 random float values to store in height_arr
        // Each value represents a sea water column height
        // At each iteration, update height_arr with a new sea water column height value
        for (int i = 0; i < MOVING_AVERAGE_WINDOW; i++)
        {
            height_arr[i] = generate_height(MIN, MAX);
        }
        
        while (termination_message == FALSE) 
        {
            // Reset every iteration
            send_reading_request = FALSE;

            // Generate and update height_arr with the new sea water column height value
            float new_height = generate_height(MIN, MAX);
            update_height_arr(height_arr, new_height);

            // Current reading (moving average)
            // Reading is obtained by calling calculate_moving_average 
            current_reading = calculate_moving_average(height_arr);

            /*
                If current reading exceeds the threshold, send requests to neighbour nodes
                Wait to receive their readings
            */

            // If current reading exceeds the threshold, send requests to neighbour nodes
            // To acquire their readings
            // Otherwise, no request sending / reading receiving is required
            if (current_reading > THRESHOLD) 
            {
                // For debugging purposes
                // printf("|Node %d| Current reading exceeds threshold: %f\n", rank, current_reading);
                send_reading_request = TRUE;
                for (int i = 0; i < MAX_NBR_COUNT; i++) 
                {
                    if (nbr_ranks[i] >= 0) 
                    {
                        // Synchronous blocking send, compared to MPI_Send
                        // MPI_Ssend can help verify code correctness
                        MPI_Ssend(&send_reading_request, 1, MPI_INT, nbr_ranks[i], NBR_READING_REQUEST, cart_comm);

                        // A MPI_Waitall() will be called later on
                        MPI_Irecv(&nbr_readings[i], 1, MPI_FLOAT, nbr_ranks[i], NBR_READING_RESPONSE, cart_comm, (nbr_responses + alloc));
                        
                        alloc++;
                        messages_passed++;
                    }
                }
                alloc = 0;
            }

            /*
                If threshold is not exceeded, check if there are requests sent from neighbour nodes
                Be prepared to send own reading to neighbour nodes
            */

            // A node's reading may be required by one or more neighbour nodes
            // MPI_Testsome is the appropriate function to use
            // It checks if one or more of the non-blocking routines pointed is complete
            MPI_Testsome(nbr_count, nbr_requests, &nbr_requests_count, completed_nbr_requests, MPI_STATUSES_IGNORE);

            // If there is one or more requests sent from neighbour nodes
            // Send own reading to them
            if (nbr_requests_count > 0) 
            {
                for (int i = 0; i < nbr_requests_count; i++) 
                {
                    // Synchronous blocking send, compared to MPI_Send
                    // MPI_Ssend can help verify code correctness
                    MPI_Ssend(&current_reading, 1, MPI_FLOAT, valid_nbr_ranks[completed_nbr_requests[i]], NBR_READING_RESPONSE, cart_comm);

                    messages_passed++;

                    // This allows the node to be prepared to receive subsequent requests from its neighbour nodes
                    MPI_Irecv(&reading_request, 1, MPI_INT, valid_nbr_ranks[completed_nbr_requests[i]], NBR_READING_REQUEST, cart_comm, (nbr_requests + completed_nbr_requests[i]));
                }
            }
            
            /*
                Wait to receive readings from neighbour nodes
            */

            // If node has sent requests to neighbour nodes
            if (send_reading_request == TRUE) 
            {
                // Wait until readings from all neighbour nodes have been received
                MPI_Waitall(nbr_count, nbr_responses, MPI_STATUSES_IGNORE);

                for (int i = 0; i < MAX_NBR_COUNT; i++) 
                {
                    float diff = nbr_readings[i] - current_reading;
                    if ((nbr_readings[i] != 0) && (diff <= (float) TOLERANCE_RANGE) && (diff >= (float) -TOLERANCE_RANGE))
                    {
                        nbr_matches++;
                    }
                }

                /*
                    Fill up the report to be sent to base station
                */

                if (nbr_matches >= 2) 
                {
                    // Alert reported time
                    report.reported_time = (long int) time(NULL);

                    // Communication time
                    double start = MPI_Wtime();
                    report.comm_time = start;

                    // Height
                    report.current_reading = current_reading;

                    // Neighbour nodes' ranks
                    for (int i = 0; i < MAX_NBR_COUNT; i++) 
                    {
                        report.nbr_ranks[i] = nbr_ranks[i];
                    }

                    // Neighbour nodes' readings
                    for (int i = 0; i < MAX_NBR_COUNT; i++)
                    {
                        report.nbr_readings[i] = nbr_readings[i];
                    }

                    // Own node's coordinates
                    report.coordinates[0] = coordinates[0];
                    report.coordinates[1] = coordinates[1];

                    // Number of reading matches
                    report.nbr_matches = nbr_matches;

                    // Synchronous blocking send, compared to MPI_Send
                    // MPI_Ssend can help verify code correctness
                    MPI_Ssend(&report, 1, report_type, bs_rank, SEND_TO_BS, MPI_COMM_WORLD);

                    // For debugging purposes
                    // printf("[Node %d| Current reading: %f, number of matches: %d\n", rank, current_reading, nbr_matches);
                    // printf("|Node %d| Sending report to base station\n", rank);
                }

                // Reset every iteration
                nbr_matches = 0;
            }

            // MPI_Iprobe allows node to always be ready to receive a termination message from the base station
            MPI_Iprobe(bs_rank, BS_TAG, MPI_COMM_WORLD, &termination_message, MPI_STATUS_IGNORE);
            
            // Interval of 0.01 seconds
            usleep(10000);
        }

        // This is just to ensure that all the readings are properly sent out
        // Acts as a satefy net to prevent nodes from waiting forever
        for (int iteration = 0; iteration < RESOLVE; iteration++) 
        {
            MPI_Testsome(nbr_count, nbr_requests, &nbr_requests_count, completed_nbr_requests, MPI_STATUSES_IGNORE);

            // If there is one or more requests sent from neighbour nodes
            // Send own reading to them
            if (nbr_requests_count > 0) 
            {
                for (int i = 0; i < nbr_requests_count; i++) 
                {
                    // Synchronous blocking send, compared to MPI_Send
                    // MPI_Ssend can help verify code correctness
                    MPI_Ssend(&current_reading, 1, MPI_FLOAT, valid_nbr_ranks[completed_nbr_requests[i]], NBR_READING_RESPONSE, cart_comm);

                    messages_passed++;

                }
            }
        }

        // For debugging purposes
        // printf("|Node %d| Base station has requested for termination\n", rank);

        // Number of messages passed while a node is active (not terminated)
        // Used for summary at the end of the log file
        MPI_Reduce(&messages_passed, &total_messages_passed, 1, MPI_INT, MPI_SUM, bs_rank, MPI_COMM_WORLD);
        
        // Deallocate memory 
        free(valid_nbr_ranks);
        free(nbr_requests);
        free(nbr_responses);
        free(completed_nbr_requests);
    } 

    /*
        Code for base station starts here
    */
    
    if (rank == bs_rank)
    {
        // Number of iterations
        int iteration_count = 0;

        // Log file to be written
        FILE *logfile;
        logfile = fopen("bs_log_file.txt", "w");

        // Total communication time
        // Used for summary at the end of the log file
        double total_comm_time;

        // Total number of true alerts occurred 
        // Used for summary at the end of the log file
        int true_alert_count = 0;

        // Total number of false alerts occurred 
        // Used for summary at the end of the log file
        int false_alert_count = 0;

        // Store tag of MPI_Iprobe which is either true (1) or false (0)
        // If true, it means that the particular node has sent a report to the base station
        int *nodes_report_tag;
        nodes_report_tag = (int *) malloc((row * col) * sizeof(int));

        // By default, MAX_ITERATION is the maximum number of iterations
        int max_iteration = MAX_ITERATION;

        // Allow input for iteration
        if (argc == 4)
        {
            max_iteration = atoi(argv[3]);
        }

        while (iteration_count < max_iteration + RESOLVE) 
        {
            // MPI_Iprobe allows base station to always be ready to receive a report from any node
            for (int i = 0; i < bs_rank; i++)
            {
                MPI_Iprobe(i, SEND_TO_BS, MPI_COMM_WORLD, &nodes_report_tag[i], MPI_STATUSES_IGNORE);
            }
            usleep(5000);

            /*
                Implement a clear and readable entry in the log file
            */

            for (int i = 0; i < bs_rank; i++) 
            {
                // If base station has received a report from a node
                if (nodes_report_tag[i] == 1) 
                {
                    // For debugging purposes
                    // printf("<Base Station> Report sent from node: %d\n", i);

                    // Properly receive the report sent by the node 
                    MPI_Recv(&report, 1, report_type, i, SEND_TO_BS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    // Time taken for the node to alert the base station
                    // From sending the report to the base station till the base station receiving it
                    double end = MPI_Wtime();
                    double comm_time = end - report.comm_time;

                    // Satellite entry's reporting time
                    time_t entry_time_recorded;

                    // Satellite entry's reading
                    float entry_reading;

                    // Satellite entry's coordinates
                    int entry_coordinates[2] = {0, 0}; 

                    // For all the entries in the satellite altimeter
                    // Check if any entry has the same coordinates as the node's own coordinates
                    // If there is a match, alert type is set to true
                    int node_satellite_match = FALSE;
                    for (int i = 0; i < (row * col); i++)
                    {
                        // If there is a match, obtain the first occurrence of entry that has matching coordinates
                        if ((satellite[i].entry_coordinates[0] == report.coordinates[0]) && (satellite[i].entry_coordinates[1] == report.coordinates[1]))
                        {
                            node_satellite_match = TRUE;

                            // Satellite entry's reporting time
                            entry_time_recorded = satellite[i].entry_time_recorded;

                            // Satellite entry's reading
                            entry_reading = satellite[i].entry_reading;

                            // Satellite entry's coordinates               
                            for (int j = 0; j < 2; j++)
                            {
                                entry_coordinates[j] = satellite[i].entry_coordinates[j];
                            } 
                            
                            break;
                        }
                    }

                    total_comm_time = total_comm_time + comm_time;

                    fprintf(logfile, "Iteration           : %d\n", iteration_count);

                    time_t logged_time;
                    time(&logged_time);                   
                    fprintf(logfile, "Logged time         : %s", ctime(&logged_time));

                    fprintf(logfile, "Alert reported time : %s", ctime((time_t *) &report.reported_time));

                    // Check if node's own reading matches with entry's reading
                    if (node_satellite_match == TRUE)
                    {        
                        float diff = entry_reading - report.current_reading;
                        if ((diff <= (float) TOLERANCE_RANGE) && (diff >= (float) -TOLERANCE_RANGE))
                        {
                            fprintf(logfile, "Alert type          : True\n\n");
                            true_alert_count++;
                        }
                        else
                        {
                            fprintf(logfile, "Alert type          : False\n\n");
                            false_alert_count++;
                        }
                    }
                    else
                    {
                        fprintf(logfile, "Alert type          : False\n\n");
                        false_alert_count++;
                    }

                    fprintf(logfile, "Reporting Node          Coord          Height (m)\n");

                    fprintf(logfile, "%d                       (%d, %d)         %.3f\n\n", i, report.coordinates[0], report.coordinates[1], report.current_reading);
                    
                    fprintf(logfile, "Adjacent Nodes          Coord          Height (m)\n");

                    // A node's report only contains its own coordinates
                    // The coordinates of its neighbour nodes can be obtained by using its own coordinates
                    //  It is simply obtained by adding or subtracting 1, on either the first or second coordinate
                    for (int i = 0; i < MAX_NBR_COUNT; i++) 
                    {
                        if (report.nbr_ranks[i] >= 0) 
                        {
                            // Top neighbour node
                            if (i == 0)
                            {
                                int top = report.coordinates[0] - 1;
                                fprintf(logfile, "%d                       (%d, %d)         %.3f\n", report.nbr_ranks[i], top, report.coordinates[1], report.nbr_readings[i]);
                            }
                            // Bottom neighbour node
                            else if (i == 1)
                            {
                                int btm = report.coordinates[0] + 1;
                                fprintf(logfile, "%d                       (%d, %d)         %.3f\n", report.nbr_ranks[i], btm, report.coordinates[1], report.nbr_readings[i]);
                            }
                            // Left neighbour node
                            else if (i == 2)
                            {
                                int left = report.coordinates[1] - 1;
                                fprintf(logfile, "%d                       (%d, %d)         %.3f\n", report.nbr_ranks[i], report.coordinates[0], left, report.nbr_readings[i]);
                            }
                            // Right neighbour node
                            else if (i == 3)
                            {
                                int right = report.coordinates[1] + 1;
                                fprintf(logfile, "%d                       (%d, %d)         %.3f\n", report.nbr_ranks[i], report.coordinates[0], right, report.nbr_readings[i]);
                            }
                        }
                    }

                    fprintf(logfile, "\n");

                    if (node_satellite_match == FALSE)
                    {
                        fprintf(logfile, "Satellite altimeter reporting time       : N/A\n");

                        fprintf(logfile, "Satellite altimeter reporting height (m) : N/A\n");
                    
                        fprintf(logfile, "Satellite altimeter reporting coord      : N/A\n\n");
                    }
                    else
                    {
                        fprintf(logfile, "Satellite altimeter reporting time       : %s", ctime(&entry_time_recorded));

                        fprintf(logfile, "Satellite altimeter reporting height (m) : %.3f\n", entry_reading);

                        fprintf(logfile, "Satellite altimeter reporting coord      : (%d, %d)\n\n", entry_coordinates[0], entry_coordinates[1]);
                    }

                    fprintf(logfile, "Communication time (seconds)                                : %.3f\n", comm_time);

                    fprintf(logfile, "Total messages sent between reporting node and base station : %d\n", node_bs_message_count);
                    
                    fprintf(logfile, "Number of adjacent matches to reporting node                : %d\n", report.nbr_matches);

                    fprintf(logfile, "Max. tolerance range between:\n");
                    
                    fprintf(logfile, "-> Node's readings (m)                                      : %d\n", TOLERANCE_RANGE);
                    
                    fprintf(logfile, "-> Satellite altimeter and reporting node readings (m)      : %d\n", TOLERANCE_RANGE);

                    fprintf(logfile, "---------------------------------------------------------------------------\n");
                }
            }

            // The RESOLVE is added to MAX_ITERATION to prevent any node from ending too early
            // Acts as a satefy net to prevent base station from hanging indefinitely
            if (iteration_count == max_iteration) 
            {
                termination_message = TRUE;
                running = FALSE;
                for (int i = 0; i < rank; i++) 
                {
                    MPI_Send(&termination_message, 1, MPI_INT, i, BS_TAG, MPI_COMM_WORLD);
                }
            }

            // If iteration_count has yet to reach MAX_ITERATION
            iteration_count++;

            // Interval of 1 second
            sleep(1);
        }

        MPI_Reduce(&messages_passed, &total_messages_passed, 1, MPI_INT, MPI_SUM, bs_rank, MPI_COMM_WORLD);

        fprintf(logfile, "Summary\n");

        fprintf(logfile, "Total number of messages passed throughout the network : %d\n", total_messages_passed);

        fprintf(logfile, "Total communication time (seconds)                     : %.3f\n", total_comm_time);

        fprintf(logfile, "Total number of true alerts                            : %d\n", true_alert_count);

        fprintf(logfile, "Total number of false alerts                           : %d\n", false_alert_count);

        // Wait for the satellite altimeter POSIX thread to terminate properly
        pthread_join(pt_id, NULL);

        // Deallocate memory
        free(nodes_report_tag);

        printf("Program has stop succesfully\n");
        printf("An output log file has been created\n");
    }
    MPI_Finalize();
    return 0;
}

// Function that represents the satellite altimeter
void *satellite_altimeter() 
{
    // The shared global array is an array of structures
    // The size is always row * col
    satellite = (struct satellite_entry *) malloc((row * col) * sizeof(struct satellite_entry));
    
    // Row and column coordinates
    int r = 0;
    int c = 0;

    // Initially, fill up the shared global array with initial entries
    // Each struct represents an entry
    for (int i = 0; i < (row * col); i++)
    {
        float new_height = generate_height(SATELLITE_MIN, SATELLITE_MAX);
        struct satellite_entry new_entry;
        new_entry.entry_time_recorded = time(NULL);
        new_entry.entry_reading = new_height;

        r = (rand() % row);
        c = (rand() % col);
        new_entry.entry_coordinates[0] = r;
        new_entry.entry_coordinates[1] = c;

        satellite[i] = new_entry;
    }

    // Generate and update satellite with the new sea water column height value
    float new_height = generate_height(SATELLITE_MIN, SATELLITE_MAX);

    while (running == TRUE) 
    {
        // Interval of 5 seconds
        sleep(5);

        // Create a new entry and update satellite periodically
        struct satellite_entry new_entry;
        new_entry.entry_time_recorded = time(NULL);
        new_entry.entry_reading = new_height;

        r = (rand() % row);
        c = (rand() % col);
        new_entry.entry_coordinates[0] = r;
        new_entry.entry_coordinates[1] = c;

        // Works the same as update_height_arr        
        for (int i = 0; i < ((row * col) - 1); i++)
        {
            satellite[i] = satellite[i + 1];
        }
        satellite[(row * col) - 1] = new_entry;

        // For debugging purposes
        // for (int i = 0; i < (row * col); i++)
        // {
        //     printf("Entry %d: Satellite reading: %f; Satellite coord (%d, %d)\n", i, satellite[i].entry_reading, satellite[i].entry_coordinates[0], satellite[i].entry_coordinates[1]);
        // }
    }

    free(satellite);
    return NULL;
}

// Function to generate a sea water column height value
float generate_height(float min, float max)
{
    // Scale = [0, 1.0]
    float scale = rand() / (float) RAND_MAX;
    
    // Formula to calculate sea water column height value
    float height = min + scale * (max - min);
    return height;
}

// Function to add a new sea water column height value into height_arr
// The last element in the array is updated with the newly added value
// Used with calculate_moving_average to generate a simple moving average
void update_height_arr(float height_arr[], float new_height)
{
    for (int i = 0; i < MOVING_AVERAGE_WINDOW - 1; i++)
    {
        height_arr[i] = height_arr[i + 1];
    }
    height_arr[MOVING_AVERAGE_WINDOW - 1] = new_height;
    return;
}

// Function to loop through height_arr to obtain the average
float calculate_moving_average(float height_arr[])
{
    float avg = 0;
    for (int i = 0; i < MOVING_AVERAGE_WINDOW; i++)
    {
        avg = avg + height_arr[i];
    }
    avg = avg / MOVING_AVERAGE_WINDOW;
    return avg;
}
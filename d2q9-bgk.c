/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <mm_malloc.h>
#include <openacc.h>
//#include "/home/ethan/spack/opt/spack/linux-fedora34-zen2/gcc-11.2.1/nvhpc-22.3-ctyx2wavppxsp23amigwhzkb4r66nqnt/Linux_x86_64/22.3/compilers/include/openacc.h"
//#include "/home/ethan/spack/opt/spack/linux-fedora34-zen2/gcc-11.2.1/intel-oneapi-mpi-2021.5.1-2nkvnbb3jw7cj3kgswm2f2q3zy72tb2n/mpi/2021.5.1/include/mpi.h"
#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"


/* struct to hold the parameter values */
typedef struct{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  int newheight;
  int cellsindex;
  int nextindex;
  int previndex;
  int world_rank;
} t_param;

/* struct to hold the 'speed' values */
typedef struct{
  float *speeds[NSPEEDS];
} t_speed;

int initialise(const char* paramfile, const char* obstaclefile,t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,int** obstacles_ptr, float** av_vels_ptr);
inline void timestep(const t_param params, t_speed* __restrict__ cells, t_speed* __restrict__ tmp_cells, const int* __restrict__ obstacles, float* __restrict__ av_vels,const float* buffers[4]);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);
/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,int** obstacles_ptr, float** av_vels_ptr);
/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);
/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);
/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);
/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
int main(int argc, char* argv[]){
  MPI_Init(NULL, NULL);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
 
  
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  /* parse the command line */
  if (argc != 3){
    usage(argv[0]);
  }
  else{
    paramfile = argv[1];
    obstaclefile = argv[2];
  }


  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);
  

  //split here

  int displacements[world_size];
  int counts[world_size];
  int quotient,remainder,add,newheight,cellsindex;
  for(int rank=0;rank<world_size;rank++){
  quotient=params.ny/world_size;
  remainder=params.ny%world_size;
  add=(rank<remainder)?1:0;
  newheight=quotient+add;
  cellsindex;
  if(add) cellsindex=newheight*rank*params.nx;
  else cellsindex=(quotient+1)*remainder*params.nx + (rank-remainder)*quotient*params.nx;
  displacements[rank]=cellsindex;
  counts[rank]=newheight*params.nx;
}


  //int newheight=params.ny/world_size;
  quotient=params.ny/world_size;
  remainder=params.ny%world_size;
  add=(world_rank<remainder)?1:0;
  newheight=quotient+add;

  if(add){
    cellsindex=newheight*world_rank*params.nx;
  }
  else{
    cellsindex=(quotient+1)*remainder*params.nx + (world_rank-remainder)*quotient*params.nx;
  }

  params.newheight=newheight;
  params.cellsindex=cellsindex;
  params.nextindex=(world_rank+1)%world_size;
  params.previndex=((world_rank-1)+world_size)%world_size;
  params.world_rank=world_rank;


  t_speed* newcells=_mm_malloc(sizeof(t_speed),64);
  t_speed* newtmpcells=_mm_malloc(sizeof(t_speed),64);
  int* newobst=_mm_malloc(sizeof(int)*counts[world_rank],64);
  memcpy(newobst,&obstacles[cellsindex],params.nx*newheight*sizeof(int));

  for(int i=0;i<NSPEEDS;i++){
    newcells->speeds[i]=_mm_malloc(sizeof(float)*params.nx*(newheight+2),64);
    newtmpcells->speeds[i]=_mm_malloc(sizeof(float)*params.nx*(newheight+2),64);
    memcpy(&(newcells->speeds[i][params.nx]),&(cells->speeds[i][cellsindex]),params.nx*newheight*sizeof(float));
  }


const float* packedsendbottom=_mm_malloc(NSPEEDS*params.nx*sizeof(float),64);
const float* packedsendtop=_mm_malloc(NSPEEDS*params.nx*sizeof(float),64);
const float* packedrecvbottom=_mm_malloc(NSPEEDS*params.nx*sizeof(float),64);
float* packedrecvtop=_mm_malloc(NSPEEDS*params.nx*sizeof(float),64);
const float* buffers[4]={packedsendbottom,packedsendtop,packedrecvbottom,packedrecvtop};


  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;


  //computation
    timestep(params, newcells, newtmpcells, newobst,av_vels,buffers);
  
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 
  



for(int i=0;i<NSPEEDS;i++){
  MPI_Gatherv(&(newcells->speeds[i][params.nx]),params.nx*newheight,MPI_FLOAT,&(cells->speeds[i][0]),&counts,&displacements,MPI_FLOAT,0,MPI_COMM_WORLD);
}

  



float* newavvels = _mm_malloc(params.maxIters*2*sizeof(float),64);
MPI_Reduce(av_vels,newavvels,params.maxIters*2,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);



if(world_rank==0){
  #pragma omp parallel for simd
  for(int i=0;i<params.maxIters;i++){
  av_vels[i]=newavvels[i*2]/newavvels[(i*2)+1];
}
  



  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, cells, obstacles, av_vels);
}

  _mm_free(newavvels);

  for(int i=0;i<NSPEEDS;i++){
    _mm_free(newcells->speeds[i]);
    _mm_free(newtmpcells->speeds[i]);
  }
    _mm_free(newcells);
  _mm_free(newtmpcells);

  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  MPI_Finalize();
  return EXIT_SUCCESS;
}




inline void timestep(const t_param params, t_speed* __restrict__ cells, t_speed* __restrict__ tmp_cells, const int* __restrict__ obstacles, float* __restrict__ av_vels,const float* buffers[4])
{


const float* packedsendbottom=buffers[0];
  const float* packedsendtop=buffers[1];
  const float* packedrecvbottom=buffers[2];
  const float* packedrecvtop=buffers[3];



#pragma acc data create(packedsendtop[:NSPEEDS*params.nx],packedsendbottom[:NSPEEDS*params.nx],packedrecvbottom[:NSPEEDS*params.nx],packedrecvtop[:NSPEEDS*params.nx]) copyin(obstacles[:params.nx*params.newheight]) copy(cells->speeds[0:NSPEEDS][0:(params.newheight+2)*params.nx]) create(tmp_cells->speeds[0:NSPEEDS][0:(params.newheight+2)*params.nx])
for (int tt = 0; tt < params.maxIters; tt++){




  



//av_vels vars
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u= 0.f;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
 


  


  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;


  /* modify the 2nd row of the grid */



  const int newheight=params.newheight;
  const int cellsindex=params.cellsindex;
  const int cellstartrow=cellsindex/params.nx;
  const int cellfinishrow=cellstartrow+newheight;
  const int nextindex=params.nextindex;
  const int previndex=params.previndex;



  //printf("rank: %d cellstartrow: %d cell endrow: %d\n",world_rank,cellstartrow,cellfinishrow);
  if((cellstartrow<=(params.ny-2)) && ((params.ny-2)<cellfinishrow)){
      const int ji = (params.ny-2)-cellstartrow+1; 



  #pragma acc parallel loop gang async present(cells,obstacles) async
  for (int ii = 0; ii < params.nx; ii++){
    
    /* if the cell is not occupied and
    ** we don't send a negative density */


    if (!obstacles[ii + (ji-1)]
        && (cells->speeds[3][ii + ji*params.nx] - w1) > 0.f
        && (cells->speeds[6][ii + ji*params.nx] - w2) > 0.f
        && (cells->speeds[7][ii + ji*params.nx] - w2) > 0.f)
    {
     // printf("actuallydoingsomething\n");
      /* increase 'east-side' densities */
      cells->speeds[1][ii + ji*params.nx] += w1;
      cells->speeds[5][ii + ji*params.nx] += w2;
      cells->speeds[8][ii + ji*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speeds[3][ii + ji*params.nx] -= w1;
      cells->speeds[6][ii + ji*params.nx] -= w2;
      cells->speeds[7][ii + ji*params.nx] -= w2;
    }
  }}





//#pragma acc update host(cells->speeds[:NSPEEDS][:params.nx*(params.newheight+2)])// async




int position1=0;
int position2=0;
  
   


   

    for(int kk=0;kk<NSPEEDS;kk++){
      acc_memcpy(&(packedsendbottom[kk*params.nx]),&(cells->speeds[kk][params.nx]),sizeof(float)*params.nx);
      acc_memcpy(&(packedsendtop[kk*params.nx]),&(cells->speeds[kk][params.nx*(newheight)]),sizeof(float)*params.nx);
      
    }

      #pragma acc host_data use_device(packedsendbottom,packedrecvtop,packedsendtop,packedrecvbottom)
      {
      MPI_Sendrecv(&(packedsendbottom[0]),params.nx*NSPEEDS,MPI_FLOAT,previndex,0,&(packedrecvtop[0]),params.nx*NSPEEDS,MPI_FLOAT,nextindex,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      MPI_Sendrecv(&(packedsendtop[0]),params.nx*NSPEEDS,MPI_FLOAT,nextindex,0,&(packedrecvbottom[0]),params.nx*NSPEEDS,MPI_FLOAT,previndex,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }

for(int kk=0;kk<NSPEEDS;kk++){
      acc_memcpy(&(cells->speeds[kk][params.nx*(newheight+1)]),&(packedrecvtop[params.nx*kk]),sizeof(float)*params.nx);
      acc_memcpy(&(cells->speeds[kk][0]),&(packedrecvbottom[params.nx*kk]),sizeof(float)*params.nx);
    }

#pragma acc wait

//#pragma acc update device(cells->speeds[0:NSPEEDS][0:params.nx*(2+params.newheight)])

 
  /* loop over _all_ cells */
  

  //#pragma omp parallel for simd reduction(+:tot_u,tot_cells)
  //#pragma omp parallel for reduction(+:tot_u,tot_cells)
  #pragma acc parallel loop gang reduction(+:tot_u,tot_cells) copy(tot_u,tot_cells)  present(cells,obstacles,tmp_cells) collapse(2) async
  for (int jj = 1; jj < (newheight)+1; jj++){
    for (int ii = 0; ii < params.nx; ii++){
      const int y_s = jj-1;
    const int y_n = jj+1;

      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      
      
    const int x_e=((ii+1)==params.nx)?0:ii+1;
    //const int x_e=(ii+1)%params.nx;
    const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      //__assume_aligned(&rearrange,64);
      float rearrange[NSPEEDS] __attribute__ ((aligned (64)));
      rearrange[0] = cells->speeds[0][ii + jj*params.nx]; /* central cell, no movement */
      rearrange[1] = cells->speeds[1][x_w + jj*params.nx]; /* east */
      rearrange[2] = cells->speeds[2][ii + y_s*params.nx]; /* north */
      rearrange[3] = cells->speeds[3][x_e + jj*params.nx]; /* west */
      rearrange[4] = cells->speeds[4][ii + y_n*params.nx]; /* south */
      rearrange[5] = cells->speeds[5][x_w + y_s*params.nx]; /* north-east */
      rearrange[6] = cells->speeds[6][x_e + y_s*params.nx]; /* cnorth-west */
      rearrange[7] = cells->speeds[7][x_e + y_n*params.nx]; /* south-west */
      rearrange[8] = cells->speeds[8][x_w + y_n*params.nx]; /* south-east */

       /* if the cell contains an obstacle */
       
      if (obstacles[(jj-1)*params.nx + ii]){
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells->speeds[0][ii + jj*params.nx] = rearrange[0];
        tmp_cells->speeds[1][ii + jj*params.nx] = rearrange[3];
        tmp_cells->speeds[2][ii + jj*params.nx] = rearrange[4];
        tmp_cells->speeds[3][ii + jj*params.nx] = rearrange[1];
        tmp_cells->speeds[4][ii + jj*params.nx] = rearrange[2];
        tmp_cells->speeds[5][ii + jj*params.nx] = rearrange[7];
        tmp_cells->speeds[6][ii + jj*params.nx] = rearrange[8];
        tmp_cells->speeds[7][ii + jj*params.nx] = rearrange[5];
        tmp_cells->speeds[8][ii + jj*params.nx] = rearrange[6];
      }



       /* don't consider occupied cells */
      else{
        const float c_sq = 1.f / 3.f; /* square of speed of sound */
        const float w0 = 4.f / 9.f;  /* weighting factor */
        const float w1 = 1.f / 9.f;  /* weighting factor */
        const float w2 = 1.f / 36.f; /* weighting factor */
        /* compute local density total */
        float local_density = 0.f;
        #pragma acc loop reduction(+:local_density)
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += rearrange[kk];
        }

        /* compute x velocity component */
        
        float u_x = (rearrange[1]+ rearrange[5]+ rearrange[8]- (rearrange[3]+ rearrange[6]+ rearrange[7]))/ local_density;
        /* compute y velocity component */
        
        float u_y = (rearrange[2]+ rearrange[5]+ rearrange[6]- (rearrange[4]+ rearrange[7] + rearrange[8]))/ local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        const float u[NSPEEDS-1] = {u_x,u_y,-u_x,-u_y,u_x+u_y,-u_x+u_y,-u_x-u_y,u_x-u_y};

        float d_equ[NSPEEDS] __attribute__ ((aligned (64)));
        d_equ[0] = w0 * local_density* (1.f - u_sq / (2.f * c_sq));
        d_equ[1] = w1 * local_density * (1.f + u[0] / c_sq+ (u[0] * u[0]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[1] / c_sq+ (u[1] * u[1]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[2] / c_sq+ (u[2] * u[2]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[3] / c_sq+ (u[3] * u[3]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
        d_equ[5] = w2 * local_density * (1.f + u[4] / c_sq+ (u[4] * u[4]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[5] / c_sq+ (u[5] * u[5]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[6] / c_sq+ (u[6] * u[6]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[7] / c_sq+ (u[7] * u[7]) / (2.f * c_sq * c_sq)- u_sq / (2.f * c_sq));
        tmp_cells->speeds[0][ii + jj*params.nx] = rearrange[0]+ params.omega * (d_equ[0] - rearrange[0]);
        tmp_cells->speeds[1][ii + jj*params.nx] = rearrange[1]+ params.omega * (d_equ[1] - rearrange[1]);
        tmp_cells->speeds[2][ii + jj*params.nx] = rearrange[2]+ params.omega * (d_equ[2] - rearrange[2]);
        tmp_cells->speeds[3][ii + jj*params.nx] = rearrange[3]+ params.omega * (d_equ[3] - rearrange[3]);
        tmp_cells->speeds[4][ii + jj*params.nx] = rearrange[4]+ params.omega * (d_equ[4] - rearrange[4]);
        tmp_cells->speeds[5][ii + jj*params.nx] = rearrange[5]+ params.omega * (d_equ[5] - rearrange[5]);
        tmp_cells->speeds[6][ii + jj*params.nx] = rearrange[6]+ params.omega * (d_equ[6] - rearrange[6]);
        tmp_cells->speeds[7][ii + jj*params.nx] = rearrange[7]+ params.omega * (d_equ[7] - rearrange[7]);
        tmp_cells->speeds[8][ii + jj*params.nx] = rearrange[8]+ params.omega * (d_equ[8] - rearrange[8]);
          
          
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf(u_sq);
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }
  #pragma acc wait
   t_speed* placeholder=cells;
    cells=tmp_cells;
    
    tmp_cells=placeholder;
  

  
    av_vels[tt*2]=tot_u;
    av_vels[(tt*2)+1]=tot_cells;
}
  
}

float av_velocity(const t_param params, t_speed* __restrict__ cells, int* __restrict__ obstacles){
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;
       
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][ii + jj*params.nx];
        }

        /* x-component of velocity */
        float u_x = (cells->speeds[1][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[8][ii + jj*params.nx]
                      - (cells->speeds[3][ii + jj*params.nx]
                         + cells->speeds[6][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speeds[2][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[6][ii + jj*params.nx]
                      - (cells->speeds[4][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]
                         + cells->speeds[8][ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,int** obstacles_ptr, float** av_vels_ptr){
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));
  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->ny));
  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->maxIters));
  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);
  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->density));
  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->accel));
  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);
  retval = fscanf(fp, "%f\n", &(params->omega));
  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  *cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed),64);
  *tmp_cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed),64);
 
  #pragma omp parallel for 
  for(int k=0;k<NSPEEDS;k++){
    (*cells_ptr)->speeds[k]=(float*)_mm_malloc(sizeof(float)*params->nx*params->ny,64);
    (*tmp_cells_ptr)->speeds[k]=(float*)_mm_malloc(sizeof(float)*params->nx*params->ny,64);
  }

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * (params->ny * params->nx),64);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  const float w0 = params->density * 4.f / 9.f;
  const float w1 = params->density      / 9.f;
  const float w2 = params->density      / 36.f;

  #pragma omp parallel for
  for (int jj = 0; jj < params->ny; jj++){
    for (int ii = 0; ii < params->nx; ii++){
      /* centre */
      (*cells_ptr)->speeds[0][ii + jj*params->nx] = w0;
      /* axis directions */
      (*cells_ptr)->speeds[1][ii + jj*params->nx] = w1;
      (*cells_ptr)->speeds[2][ii + jj*params->nx] = w1;
      (*cells_ptr)->speeds[3][ii + jj*params->nx] = w1;
      (*cells_ptr)->speeds[4][ii + jj*params->nx] = w1;
      /* diagonals */
      (*cells_ptr)->speeds[5][ii + jj*params->nx] = w2;
      (*cells_ptr)->speeds[6][ii + jj*params->nx] = w2;
      (*cells_ptr)->speeds[7][ii + jj*params->nx] = w2;
      (*cells_ptr)->speeds[8][ii + jj*params->nx] = w2;
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL){
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF){
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)_mm_malloc(sizeof(float) * params->maxIters*2,64);
  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,int** obstacles_ptr, float** av_vels_ptr){
  /*
  ** free up allocated memory
  */
 for(int i=0;i<NSPEEDS;i++){
   _mm_free((*cells_ptr)->speeds[i]);
   _mm_free((*tmp_cells_ptr)->speeds[i]);
   
 }
  _mm_free(*cells_ptr);
  *cells_ptr = NULL;

  _mm_free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  _mm_free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles){
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells){
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells->speeds[kk][ii + jj*params.nx];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels){
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells->speeds[kk][ii + jj*params.nx];
        }

        /* compute x velocity component */
        /* x-component of velocity */
        float u_x = (cells->speeds[1][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[8][ii + jj*params.nx]
                      - (cells->speeds[3][ii + jj*params.nx]
                         + cells->speeds[6][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->speeds[2][ii + jj*params.nx]
                      + cells->speeds[5][ii + jj*params.nx]
                      + cells->speeds[6][ii + jj*params.nx]
                      - (cells->speeds[4][ii + jj*params.nx]
                         + cells->speeds[7][ii + jj*params.nx]
                         + cells->speeds[8][ii + jj*params.nx]))
                     / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}
void die(const char* message, const int line, const char* file){
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe){
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

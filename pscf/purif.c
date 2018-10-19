#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <mkl.h>
#include <mkl_trans.h>
#include <sys/time.h>
#include <ga.h>

#include "pdgemm.h"
#include "purif.h"


#define MAX_PURF_ITERS 200
#define MIN(a, b)    ((a) < (b) ? (a) : (b))
#define MAX(a, b)    ((a) > (b) ? (a) : (b))

static void config_purif(purif_t * purif, int purif_offload)
{
    int nbf, nrows, ncols, nb, nprow, npcol;
    int *nr, *nc, startrow, endrow, startcol, endcol;
    MPI_Comm comm_row, comm_col;

    nbf = purif->nbf;
    nprow = purif->nprow_purif;
    npcol = purif->npcol_purif;
    purif->nb_purif = nb = MIN (nbf / nprow, nbf / npcol);
    comm_row = purif->comm_purif_row;
    comm_col = purif->comm_purif_col;

    int coords[3];
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Cart_coords(purif->comm_purif, myrank, 3, coords);
    int myrow = coords[0];
    int mycol = coords[1];
    int mygrd = coords[2];

    int izero = 0;
    nrows = numroc_(&nbf, &nb, &myrow, &izero, &nprow);
    ncols = numroc_(&nbf, &nb, &mycol, &izero, &npcol);
    purif->nrows_purif = nrows;
    purif->ncols_purif = ncols;

    // positions of partitions
    purif->nr_purif = (int *) malloc (sizeof (int) * purif->nprow_purif);
    purif->nc_purif = (int *) malloc (sizeof (int) * purif->npcol_purif);
    assert (purif->nr_purif != NULL);
    assert (purif->nc_purif != NULL);
    nr = purif->nr_purif;
    nc = purif->nc_purif;

    // get nr and sr
    MPI_Allgather(&nrows, 1, MPI_INT, nr, 1, MPI_INT, comm_row);
    startrow = 0;
    for (int i = 0; i < myrow; i++) startrow += nr[i];
    endrow = startrow + nrows - 1;
    purif->srow_purif = startrow;

    // get nc and sc
    MPI_Allgather(&ncols, 1, MPI_INT, nc, 1, MPI_INT, comm_col);
    startcol = 0;
    for (int i = 0; i < mycol; i++) startcol += nc[i];
    endcol = startcol + ncols - 1;
    purif->scol_purif = startcol;

    // for matrix trace
    int start = MAX(startcol, startrow);
    purif->tr_len_purif  = MIN(endcol, endrow) - start + 1;
    purif->tr_scol_purif = start - startcol;
    purif->tr_srow_purif = start - startrow;
    purif->istr_purif = (purif->tr_len_purif > 0);

    // create local arrays
    purif->ldx = ncols;
    int meshsize = nrows * ncols;
    size_t mesh_memsize = meshsize * sizeof(double);
    purif->meshsize = meshsize;
    purif->X_block  = (double *) _mm_malloc(mesh_memsize, 64);
    purif->S_block  = (double *) _mm_malloc(mesh_memsize, 64);
    purif->H_block  = (double *) _mm_malloc(mesh_memsize, 64);
    purif->F_block  = (double *) _mm_malloc(mesh_memsize, 64);
    purif->D_block  = (double *) _mm_malloc(mesh_memsize, 64);
    purif->D2_block = (double *) _mm_malloc(mesh_memsize, 64);
    purif->D3_block = (double *) _mm_malloc(mesh_memsize, 64);
    assert (purif->X_block  != NULL);
    assert (purif->S_block  != NULL);
    assert (purif->H_block  != NULL);
    assert (purif->F_block  != NULL);
    assert (purif->D_block  != NULL);
    assert (purif->D2_block != NULL);
    assert (purif->D3_block != NULL);
    // working space for purification
    purif->diis_vecs = (double *) _mm_malloc (MAX_DIIS * mesh_memsize, 64);
    purif->F_vecs    = (double *) _mm_malloc (MAX_DIIS * mesh_memsize, 64);
    assert (purif->diis_vecs != NULL);
    assert (purif->F_vecs != NULL);
    purif->len_diis = 0;
    purif->bmax = DBL_MIN;
    purif->bmax_id = -1;
    assert (MAX_DIIS > 1);
    for (int i = 0; i < LDBMAT; i++)
    {
        for (int j = 0; j < LDBMAT; j++)
            purif->b_mat[i * LDBMAT + j] = -1.0;
        purif->b_mat[i * LDBMAT + i] = 0.0;
    }

    #pragma omp parallel for schedule(static)
    #pragma simd
    for(int i=0; i < nrows * ncols; i++)
    {
        purif->D_block[i]  = 0.0;
        purif->D2_block[i] = 0.0;
        purif->D3_block[i] = 0.0;
    }
    allocate_tmpbuf(nrows, ncols, nr, nc, &(purif->tmpbuf));
    size_t memsize = (2.0 * MAX_DIIS + 14.0) * meshsize * sizeof (double);
    
    purif->h  = (double*) _mm_malloc(2 * purif->ncols_purif * sizeof(double), 64);
    purif->_h = (double*) _mm_malloc(2 * purif->ncols_purif * sizeof(double), 64);
    
    
    if (myrow == 0 && mycol == 0 && mygrd == 0)
        printf("  CPU uses %.3f MB\n", memsize / 1024.0 / 1024.0);
}


purif_t *create_purif(BasisSet_t basis, int nprow_purif, int npcol_purif, int npgrd_purif)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    if (myrank == 0) printf ("Initializing purification ...\n");

    // create purif
    purif_t *purif = (purif_t *) malloc(sizeof (purif_t));
    assert (purif != NULL);
    purif->nbf = CInt_getNumFuncs(basis);
    purif->nobtls = CInt_getNumOccOrb(basis);
    purif->nprow_purif = nprow_purif;
    purif->npcol_purif = npcol_purif;
    purif->npgrd_purif = npgrd_purif;
    purif->np_purif    = nprow_purif * npcol_purif * npgrd_purif;

    // set node types
    purif->runpurif = 0;
    purif->rundgemm = (myrank < purif->np_purif) ? 1 : 0;

    // initialize communicators
    int flag_purif = (myrank < purif->np_purif);
    MPI_Comm comm0;
    MPI_Comm_split(MPI_COMM_WORLD, flag_purif, myrank, &comm0);

    
    if (purif->rundgemm == 1)
    {
        int ndims = 3, reorder = 1;
        int dim_size[3] = {nprow_purif, npcol_purif, npgrd_purif};
        int periods[3]  = {0, 0, 0};
        int coords[3];
        MPI_Cart_create(comm0, ndims, dim_size, periods, reorder, &(purif->comm_purif));
        MPI_Cart_coords(purif->comm_purif, myrank, ndims, coords);
        int myrow = coords[0];
        int mycol = coords[1];
        int mygrd = coords[2];

        purif->runpurif = (mygrd == 0) ? 1 : 0;

        int belongsR[3] = {1, 0, 0};
        int belongsC[3] = {0, 1, 0};
        int belongsG[3] = {0, 0, 1};
        int plane_rank  = myrow * npcol_purif + mycol;
        MPI_Cart_sub(purif->comm_purif, belongsR, &(purif->comm_purif_row));
        MPI_Cart_sub(purif->comm_purif, belongsC, &(purif->comm_purif_col));
        MPI_Cart_sub(purif->comm_purif, belongsG, &(purif->comm_purif_grd));
        MPI_Comm_split(purif->comm_purif, mygrd, plane_rank, &(purif->comm_purif_plane));
        config_purif(purif, 0);
    }
    
    if (myrank == 0) printf ("  Done\n");
    
    return purif;
}


void destroy_purif(purif_t * purif)
{
    if (purif->rundgemm == 1)
    {
        dealloc_tmpbuf(&(purif->tmpbuf));
        
        MPI_Comm_free(&(purif->comm_purif));
        MPI_Comm_free(&(purif->comm_purif_col));
        MPI_Comm_free(&(purif->comm_purif_row));
        free(purif->nr_purif);
        free(purif->nc_purif);
        _mm_free(purif->H_block);
        _mm_free(purif->X_block);
        _mm_free(purif->S_block);
        _mm_free(purif->F_block);
        _mm_free(purif->D_block);
        _mm_free(purif->D3_block);
        _mm_free(purif->D2_block);
        _mm_free(purif->F_vecs);
        _mm_free(purif->diis_vecs);
        _mm_free(purif->h);
        _mm_free(purif->_h);
    }
    free(purif);
}


static double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

int compute_purification(purif_t * purif, double *F_block, double *D_block)
{
    int it;
    purif->timedgemm  = 0.0;
    purif->timepdgemm = 0.0;
    purif->timepass = 0.0;
    purif->timetr = 0.0;
    double st1, et1, st2, et2;

    if (purif->rundgemm == 1) 
    {
        st1 = get_wtime_sec();
        
        // initialization
        int nrows = purif->nrows_purif;
        int ncols = purif->ncols_purif;
        int startrow = purif->srow_purif;
        int startcol = purif->scol_purif;
        int starttrrow = purif->tr_srow_purif;
        int starttrcol = purif->tr_scol_purif;
        int lentr = purif->tr_len_purif;
        int *nr = purif->nr_purif;
        int *nc = purif->nc_purif;
        MPI_Comm comm_row = purif->comm_purif_row;
        MPI_Comm comm_col = purif->comm_purif_col;
        MPI_Comm comm_grd = purif->comm_purif_grd;
        MPI_Comm comm_purif = purif->comm_purif_plane;
        MPI_Comm comm0 = purif->comm_purif;
        int nbf = purif->nbf;
        int nobtls = purif->nobtls;
        double *X_block  = purif->X_block;
        double *D2_block = purif->D2_block;
        double *D3_block = purif->D3_block;
        double *workm = D2_block;
        int coords[3];
        int myrank;
        MPI_Comm_rank(purif->comm_purif, &myrank);
        MPI_Cart_coords(purif->comm_purif, myrank, 3, coords);
        int myrow = coords[0];
        int mycol = coords[1];
        int mygrd = coords[2];
        tmpbuf_t tmpbuf = purif->tmpbuf;
        
        if (purif->runpurif == 1) 
        {
            // compute eigenvalue estimates using Gershgorin (F is symmetric)
            // offdF = sum(abs(F))' - abs(diag(F));
            // diagF = diag(Fs);
            // hmin = min(diagF - offdF);
            // hmax = max(diagF + offdF);
            double *h  = purif->h;  // hmin, hmax
            double *_h = purif->_h; // offDF, diagF
            for (int i = 0; i < ncols; i++) 
            {
                _h[i] = 0.0;
                _h[i + ncols] = 0.0;
                for (int j = 0; j < nrows; j++) 
                {
                    _h[i] += fabs(F_block[i + j * ncols]);
                    if (j + startrow == i + startcol) 
                        _h[i + ncols] = F_block[i + j * ncols];
                }
                _h[i] = _h[i] - fabs(_h[i + ncols]);
                double tmp = _h[i + ncols] + _h[i];
                _h[i] = _h[i + ncols] - _h[i];
                _h[i + ncols] = tmp;
            }
            MPI_Reduce(_h, h, 2 * ncols, MPI_DOUBLE, MPI_SUM, 0, comm_row);

            double _hmax, _hmin, hmax, hmin;
            if (myrow == 0) 
            {
                _hmin = DBL_MAX;
                _hmax = -DBL_MAX;
                for (int i = 0; i < ncols; i++) 
                {
                    _hmin = h[i] > _hmin ? _hmin : h[i];
                    _hmax = h[i + ncols] < _hmax ? _hmax : h[i + ncols];
                }
                MPI_Reduce(&_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, 0, comm_col);
                MPI_Reduce(&_hmax, &hmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_col);
            }
            MPI_Bcast(&hmin, 1, MPI_DOUBLE, 0, comm_purif);
            MPI_Bcast(&hmax, 1, MPI_DOUBLE, 0, comm_purif);
            // define constants, dependent on F
            // in the following:
            // 5 = no of occupied orbitals
            // 7 = no of spatial basis function
            // (each corresponds to 2 electrons for RHF)
            // mu_bar = trace_dense_matrix(F)/7;
            double trF, _trF;
            _trF = 0.0;
            for (int i = 0; i < lentr; i++) 
                _trF += F_block[(i + starttrrow) * ncols + i + starttrcol];
            
            MPI_Allreduce(&_trF, &trF, 1, MPI_DOUBLE, MPI_SUM, comm_purif);

            double mu_bar = trF / (double) nbf;
            // lambda = min([ 5/(hmax - mu_bar), (7-5)/(mu_bar - hmin) ]);
            double lambda = MIN((double) nobtls / (hmax - mu_bar),
                                (double) (nbf - nobtls) / (mu_bar - hmin));
            if (myrank == 0) 
            {
                printf("mu_bar = %le, lambda = %le,"
                       " hmax = %le, hmin = %le, nobtls = %d\n",
                       mu_bar, lambda, hmax, hmin, nobtls);
            }
            
            // initial "guess" for density matrix
            // D = (lambda*mu_bar/7 + 5/7)*eye(7) - (lambda/7)*D;
            for (int i = 0; i < nrows * ncols; i++) 
                D_block[i] = F_block[i] * (-lambda / nbf);
            
            for (int i = 0; i < lentr; i++) 
            {
                D_block[(i + starttrrow) * ncols + i + starttrcol] +=
                    lambda * mu_bar / (double) nbf + (double) nobtls / nbf;
            }
        } /* if (purif->runpurif == 1) */

        // McWeeny purification
        // convergence appears slow at first, before accelerating at end
        double dgemm_time, pdgemm_s, pdgemm_e, tr_s, tr_e, errnorm;
        double tr[2], tr_local[2], c, errnorm_local, n2cp1, cp1, ncp1;
        for (it = 0; it < MAX_PURF_ITERS; it++) 
        {
            pdgemm_s = get_wtime_sec();
            pdgemm3D(myrow, mycol, mygrd, comm_row, comm_col, comm_grd,
                     comm0, nr, nc, nrows, ncols, D_block, D2_block,
                     D3_block, &tmpbuf, &dgemm_time);
            pdgemm_e = get_wtime_sec();
            purif->timepdgemm += pdgemm_e - pdgemm_s;
            purif->timedgemm  += dgemm_time;

            tr_s = get_wtime_sec();
            if (purif->runpurif == 1) 
            {
                // Stopping criterion: errnorm = norm(D-D2, 'fro');
                // A cheaper stopping criterion may be to check the trace of D*D and
                // stop when it is close to no. occupied orbitals (5 in this case).
                // fprintf('trace D*D //f\n', trace(D*D));
                // Might be possible to "lag" the computation of c by one iteration
                // so that the global communication for traces can be overlapped.
                // Note: c appears to converge to 0.5           
                // c = trace(D2-D3) / trace(D-D2);
                errnorm_local = 0.0;
                tr_local[0] = 0.0;
                tr_local[1] = 0.0;
                for (int i = 0; i < lentr; i++) 
                {
                    int idx_ii = (i + starttrrow) * ncols + (i + starttrcol);
                    tr_local[0] += D2_block[idx_ii] - D3_block[idx_ii];
                    tr_local[1] +=  D_block[idx_ii] - D2_block[idx_ii];
                }
                MPI_Allreduce(tr_local, tr, 2, MPI_DOUBLE, MPI_SUM, comm_purif);
                c = tr[0] / tr[1];
                
                n2cp1 = 1.0 - 2.0 * c;
                cp1   = c + 1.0;
                ncp1  = 1.0 - c;
                if (c < 0.5) 
                {
                    #pragma omp parallel for reduction(+: errnorm_local)
                    #pragma simd
                    for (int i = 0; i < nrows * ncols; i++) 
                    {
                        double D_D2 = D_block[i] - D2_block[i];
                        errnorm_local += D_D2 * D_D2;
                        
                        // D = ((1-2*c)*D + (1+c)*D2 - D3) / (1-c);
                        D_block[i] = (n2cp1 * D_block[i] + cp1 * D2_block[i] - D3_block[i]) / ncp1;
                    }
                } else {
                    #pragma omp parallel for reduction(+: errnorm_local)
                    #pragma simd
                    for (int i = 0; i < nrows * ncols; i++) 
                    {
                        double D_D2 = D_block[i] - D2_block[i];
                        errnorm_local += D_D2 * D_D2;
                        
                        // D = ((1+c)*D2 - D3) / c;
                        D_block[i] = (cp1 * D2_block[i] - D3_block[i]) / c;
                    }
                }
                
                MPI_Reduce(&errnorm_local, &errnorm, 1, MPI_DOUBLE, MPI_SUM, 0, comm_purif);
                if (myrank == 0) errnorm = sqrt(errnorm);
            }
            MPI_Bcast(&errnorm, 1, MPI_DOUBLE, 0, comm0);
            tr_e = get_wtime_sec();
            purif->timetr += tr_e - tr_s;
            if (errnorm < 1e-11) break;
        }

        st2 = get_wtime_sec();
        double t_dgemm1, t_dgemm2;
        pdgemm3D_2(myrow, mycol, mygrd, comm_row, comm_col, comm_grd,
                   comm0, nr, nc, nrows, ncols,
                   X_block, D_block, workm, &tmpbuf, &t_dgemm1);
        pdgemm3D_2(myrow, mycol, mygrd, comm_row, comm_col, comm_grd,
                   comm0, nr, nc, nrows, ncols,
                   workm, X_block, D_block, &tmpbuf, &t_dgemm2);
        purif->timedgemm += t_dgemm1 + t_dgemm2;
        et2 = get_wtime_sec();
        purif->timepdgemm += et2 - st2;

        et1 = get_wtime_sec();
        purif->timepass += et1 - st1;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    return it;
}


void compute_diis(PFock_t pfock, purif_t * purif, double *D_block, double *F_block, int iter)
{
    int nrows = purif->nrows_purif;
    int ncols = purif->ncols_purif;
    int meshsize = purif->meshsize;
    double *X_block = purif->X_block;
    double *S_block = purif->S_block;
    double *workm = purif->D2_block;
    double *b_mat = purif->b_mat;

    double *F_vecs = purif->F_vecs;
    double *diis_vecs = purif->diis_vecs;
    int *nr = purif->nr_purif;
    int *nc = purif->nc_purif;
    int myrank;
    MPI_Comm comm_col = purif->comm_purif_col;
    MPI_Comm comm_row = purif->comm_purif_row;
    MPI_Comm comm_grd = purif->comm_purif_grd;
    MPI_Comm comm_purif = purif->comm_purif_plane;
    MPI_Comm comm0 = purif->comm_purif;
    tmpbuf_t tmpbuf = purif->tmpbuf;

    if (purif->rundgemm == 1) {    
        int coords[3];
        MPI_Comm_rank (purif->comm_purif, &myrank);
        MPI_Cart_coords (purif->comm_purif, myrank, 3, coords);
        int myrow = coords[0];
        int mycol = coords[1];
        int mygrd = coords[2];
        double *cur_F = F_block;
        int cur_idx;
        double *cur_diis;    
        if (iter > 1) {
            if (purif->len_diis < MAX_DIIS) {
                cur_idx = purif->len_diis;
                cur_diis = &(diis_vecs[cur_idx * meshsize]);
                cur_F = &(F_vecs[cur_idx * meshsize]);
                purif->len_diis++;
            } else {
                cur_idx = purif->bmax_id;
                cur_diis = &(diis_vecs[cur_idx * meshsize]);
                cur_F = &(F_vecs[cur_idx * meshsize]);
            }
            // Ctator = X*(F*D*S - S*D*F)*X;
            pdgemm3D_2(myrow, mycol, mygrd, comm_row, comm_col, comm_grd,
                       comm0, nr, nc, nrows, ncols,
                       F_block, D_block, workm, &tmpbuf, NULL);
            pdgemm3D_2(myrow, mycol, mygrd, comm_row, comm_col, comm_grd,
                       comm0, nr, nc, nrows, ncols,
                       workm, S_block, cur_diis, &tmpbuf, NULL);
            if (purif->runpurif == 1) {
                int dest;
                coords[0] = mycol;
                coords[1] = myrow;
                coords[2] = mygrd;               
                MPI_Cart_rank(purif->comm_purif, coords, &dest);
                MPI_Sendrecv(cur_diis, nrows * ncols,
                             MPI_DOUBLE, dest, dest,
                             workm, nrows * ncols,
                             MPI_DOUBLE, dest, MPI_ANY_TAG,
                             purif->comm_purif, MPI_STATUS_IGNORE);
                // F*D*S - (F*D*S)'       
                for (int i = 0; i < nrows; i++) {
                    for (int j = 0; j < ncols; j++) {
                        cur_diis[i * ncols + j] -= workm[j * nrows + i];
                    }
                }
            }
            pdgemm3D_2(myrow, mycol, mygrd, comm_row, comm_col, comm_grd,
                       comm0, nr, nc, nrows, ncols, X_block, cur_diis,
                       workm, &tmpbuf, NULL);
            pdgemm3D_2(myrow, mycol, mygrd, comm_row, comm_col, comm_grd,
                       comm0, nr, nc, nrows, ncols, workm, X_block,
                       cur_diis, &tmpbuf, NULL);
            if (purif->runpurif == 1) {
                // b_mat(i,j) = dot(vecs(:,i), vecs(:,j));
                double _dot[LDBMAT];
                for (int i = 0; i < purif->len_diis; i++) {
                    double *diis2 = &(diis_vecs[i * meshsize]);
                    _dot[i] = 0.0;
                    for (int j = 0; j < nrows; j++) {
                        for (int k = 0; k < ncols; k++) {
                            _dot[i] +=
                                cur_diis[j * ncols + k] * diis2[j * ncols + k];
                        }
                    }
                } /* end for */
                // update b_mat on rank 0          
                MPI_Reduce(_dot, &(b_mat[cur_idx * LDBMAT]),
                           purif->len_diis, MPI_DOUBLE, MPI_SUM, 0, comm_purif);
                if (myrank == 0) {
                    purif->bmax = -DBL_MAX;
                    for (int i = 0; i < purif->len_diis; i++) {
                        b_mat[i * LDBMAT + cur_idx] =
                            b_mat[cur_idx * LDBMAT + i];
                        if (purif->bmax < b_mat[i * LDBMAT + i]) {
                            purif->bmax = b_mat[i * LDBMAT + i];
                            purif->bmax_id = i;
                        }
                    }
                }
            } /* if (purif->runpurif == 1) */
            MPI_Bcast (&(purif->bmax_id), 1, MPI_DOUBLE, 0, comm0);
        } /* if (iter > 1) */

        // F = X*F*X;
        pdgemm3D_2(myrow, mycol, mygrd, comm_row, comm_col, comm_grd,
                   comm0, nr, nc, nrows, ncols,
                   X_block, F_block, workm, &tmpbuf, NULL);
        pdgemm3D_2(myrow, mycol, mygrd, comm_row, comm_col, comm_grd,
                   comm0, nr, nc, nrows, ncols, workm,
                   X_block, cur_F, &tmpbuf, NULL);
        // extrapolate
        if (iter > 1) {
            if (purif->runpurif == 1) {
                double coeffs[LDBMAT];
                // rhs = zeros(m+1,1);
                // rhs(m+1,1) = -1;
                // coeffs = inv(b_mat) * rhs;
                if (myrank == 0) {
                    int sizeb = purif->len_diis + 1;
                    __declspec(align (64)) double b_inv[LDBMAT * LDBMAT];
                    __declspec(align (64)) int ipiv[LDBMAT];
                    memcpy(b_inv, b_mat, LDBMAT * LDBMAT * sizeof (double));
                    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, sizeb, sizeb, b_inv,
                                   LDBMAT, ipiv);
                    LAPACKE_dgetri(LAPACK_ROW_MAJOR, sizeb, b_inv, LDBMAT,
                                   ipiv);
                    for (int i = 0; i < sizeb; i++) {
                        coeffs[i] = -b_inv[i * LDBMAT + sizeb - 1];
                    }
                }
                MPI_Bcast(coeffs, purif->len_diis, MPI_DOUBLE, 0, comm_purif);

                // F = 0
                // for j = 1:m
                //     F = F + coeffs(j)* F_vecs(j);
                memset(F_block, 0, sizeof (double) * meshsize);
                for (int i = 0; i < purif->len_diis; i++) {
                    double *F_vec = &(F_vecs[i * meshsize]);
                    for (int j = 0; j < meshsize; j++) {
                        F_block[j] += coeffs[i] * F_vec[j];
                    }
                }
            } /* if (purif->runpurif == 1) */
        }
    } /* if (purif->rundgemm == 1) */

    MPI_Barrier(MPI_COMM_WORLD);
}


#if 1
static void peig(int ga_A, int ga_B, int n, int nprow, int npcol, double *eval)
{
    int myrank;
    int ictxt;
    int myrow;
    int mycol;
    int nn;
    int mm;
    int izero = 0;
    int descA[9];
    int descZ[9];
    int info;
    int lo[2];
    int hi[2];
    int ld;
    int ione = 1;
#ifdef GA_NB
    ga_nbhdl_t nbnb;
#endif

    // init blacs
    int nb = MIN(n / nprow, n / npcol);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    Cblacs_pinfo(&nn, &mm);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Row", nprow, npcol);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    // init matrices
    int nrows = numroc_(&n, &nb, &myrow, &izero, &nprow);
    int ncols = numroc_(&n, &nb, &mycol, &izero, &npcol);
    int itemp = nrows > 1 ? nrows : 1;
    descinit_(descA, &n, &n, &nb, &nb, &izero, &izero, &ictxt, &itemp, &info);
    descinit_(descZ, &n, &n, &nb, &nb, &izero, &izero, &ictxt, &itemp, &info);
    int blocksize = nrows * ncols;
    double *A = (double *)_mm_malloc(blocksize * sizeof (double), 64);
    assert (A != NULL);
    double *Z = (double *)_mm_malloc(blocksize * sizeof (double), 64);
    assert (Z != NULL);

    // distribute source matrix
    for (int i = 1; i <= nrows; i += nb) {
        lo[0] = indxl2g_(&i, &nb, &myrow, &izero, &nprow) - 1;
        hi[0] = lo[0] + nb - 1;
        hi[0] = hi[0] >= n ? n - 1 : hi[0];
        for (int j = 1; j <= ncols; j += nb) {
            lo[1] = indxl2g_(&j, &nb, &mycol, &izero, &npcol) - 1;
            hi[1] = lo[1] + nb - 1;
            hi[1] = hi[1] >= n ? n - 1 : hi[1];
            ld = ncols;
#ifdef GA_NB
            NGA_NbGet(ga_A, lo, hi, &(Z[(i - 1) * ncols + j - 1]), &ld, &nbnb);
#else
            NGA_Get(ga_A, lo, hi, &(Z[(i - 1) * ncols + j - 1]), &ld);
#endif
        }
        /* Jeff: Location of NGA_NbWait for flow-control. */
    }
#ifdef GA_NB
    /* Jeff: If one sees flow-control problems with too many
     *       outstanding NbGet operations, then move this call
     *       to the location noted above. */
    NGA_NbWait(&nbnb);
#endif
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            A[j * nrows + i] = Z[i * ncols + j];
        }
    }

    double t1 = MPI_Wtime();
    // inquire working space
    double *work = (double *)_mm_malloc(2 * sizeof (double), 64);
    assert (work != NULL);
    int lwork = -1;
#if 0
    pdsyev ("V", "U", &n, A, &ione, &ione, descA,
            eval, Z, &ione, &ione, descZ, work, &lwork, &info);
#else
    int liwork = -1;
    int *iwork = (int *)_mm_malloc(2 * sizeof (int), 64);
    assert(iwork != NULL);
    pdsyevd_("V", "U", &n, A, &ione, &ione, descA,
            eval, Z, &ione, &ione, descZ,
            work, &lwork, iwork, &liwork, &info);    
#endif

    // compute eigenvalues and eigenvectors
    lwork = (int)work[0] * 2;
    _mm_free(work);
    work = (double *)_mm_malloc(lwork * sizeof (double), 64);
    assert(work != NULL);
#if 0
    pdsyev ("V", "U", &n, A, &ione, &ione, descA,
            eval, Z, &ione, &ione, descZ, work, &lwork, &info);
#else
    liwork = (int)iwork[0];
    _mm_free(iwork);
    iwork = (int *)_mm_malloc(liwork * sizeof (int), 64);
    assert(iwork != NULL);
    pdsyevd_("V", "U", &n, A, &ione, &ione, descA,
            eval, Z, &ione, &ione, descZ,
            work, &lwork, iwork, &liwork, &info); 
#endif
    double t2 = MPI_Wtime();
    if (myrank == 0) {
        printf("  pdsyev_ takes %.3f secs\n", t2 - t1);
    }

    // store desination matrix
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            A[i * ncols + j] = Z[j * nrows + i];
        }
    }
    for (int i = 1; i <= nrows; i += nb) {
        lo[0] = indxl2g_ (&i, &nb, &myrow, &izero, &nprow) - 1;
        hi[0] = lo[0] + nb - 1;
        hi[0] = hi[0] >= n ? n - 1 : hi[0];
        for (int j = 1; j <= ncols; j += nb) {
            lo[1] = indxl2g_ (&j, &nb, &mycol, &izero, &npcol) - 1;
            hi[1] = lo[1] + nb - 1;
            hi[1] = hi[1] >= n ? n - 1 : hi[1];
            ld = ncols;
#ifdef GA_NB
            NGA_NbPut(ga_B, lo, hi, &(A[(i - 1) * ncols + j - 1]), &ld, &nbnb);
#else
            NGA_Put(ga_B, lo, hi, &(A[(i - 1) * ncols + j - 1]), &ld);
#endif
        }
        /* Jeff: Location of NGA_NbWait for flow-control. */
    }
#ifdef GA_NB
    /* Jeff: If one sees flow-control problems with too many
     *       outstanding NbPut operations, then move this call
     *       to the location noted above. */
    NGA_NbWait(&nbnb);
#endif
    GA_Sync();

    _mm_free(A);
    _mm_free(Z);
    _mm_free(work);

    Cblacs_gridexit(ictxt);
}


void compute_eigensolve(int ga_tmp, purif_t * purif, double *F_block, int nprow, int npcol)
{
    // edmond
    int myga  = GA_Duplicate(ga_tmp, "edmond mat");
    int myga2 = GA_Duplicate(ga_tmp, "edmond mat2");

    int nbf = purif->nbf;
    if (purif->runpurif == 1) {
        int lo[2];
        int hi[2];
        int ld;
        lo[0] = purif->srow_purif;
        hi[0] = purif->srow_purif + purif->nrows_purif - 1;
        lo[1] = purif->scol_purif;
        hi[1] = purif->scol_purif + purif->ncols_purif - 1;
        ld = purif->ldx;
        NGA_Put(myga, lo, hi, F_block, &ld);
    }

    double *eval = (double *)_mm_malloc(nbf * sizeof (double), 64);
    assert(eval != NULL);
    peig(myga, myga2, nbf, nprow, npcol, eval);

    // edmond
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);    
    if (myrank == 0)
    {
        int i;
        for (i=0; i<nbf; i++)
            printf("%5d    %.15e\n", i+1, eval[i]);
    }

    GA_Destroy(myga);
    GA_Destroy(myga2);
    _mm_free(eval);
}

#endif

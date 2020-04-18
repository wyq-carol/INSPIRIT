import Libdl
using StarPU
using LinearAlgebra

@target STARPU_CPU+STARPU_CUDA
@codelet function mandelbrot(pixels ::Matrix{Int64}, params ::Matrix{Float32} ) :: Nothing
    height :: Int64 = height(pixels)
    width :: Int64 = width(pixels)
    zoom :: Float64 = width * 0.25296875
    iz :: Float64 = 1. / zoom
    diverge :: Float32 = 4.0
    max_iterations :: Float32 = ((width/2) * 0.049715909 * log10(zoom));
    imi :: Float32 = 1. / max_iterations
    centerr :: Float32 = params[1,1]
    centeri :: Float32 = params[2,1]
    offset :: Float32 = params[3,1]
    dim :: Float32 = params[4,1]
    cr :: Float64 = 0.
    zr :: Float64 = 0.
    ci :: Float64 = 0.
    zi :: Float64 = 0.
    n :: Int64 = 0
    tmp :: Float64 = 0.
    @parallel for y = 1:height
        for x = 1:width
            cr = centerr + (x-1 - (dim / 2)) * iz
            zr = cr
            ci = centeri + (y-1+offset - (dim / 2)) * iz
            zi = ci
            for n = 0:max_iterations
                if (zr*zr + zi*zi > diverge)
                    break
                end
                tmp = zr*zr - zi*zi + cr
                zi = 2*zr*zi + ci
                zr = tmp
            end
            
            if (n < max_iterations)
                pixels[y,x] = round(15 * n * imi)
            else
                pixels[y,x] = 0
            end
        end
    end

    return
end

@debugprint "starpu_init"
starpu_init()

function mandelbrot_with_starpu(A ::Matrix{Int64}, params ::Matrix{Float32}, nslicesx ::Int64)
    horiz = StarpuDataFilter(STARPU_MATRIX_FILTER_BLOCK, nslicesx)
    @starpu_block let
	hA, hP = starpu_data_register(A,params)
	starpu_data_partition(hA,horiz)
        starpu_data_partition(hP,horiz)
        
	@starpu_sync_tasks for taskx in (1 : nslicesx)
                @starpu_async_cl mandelbrot(hA[taskx], hP[taskx]) [STARPU_W, STARPU_R]
	end
    end
end

function pixels2img(pixels ::Matrix{Int64}, width ::Int64, height ::Int64, filename ::String)
    MAPPING = [[66,30,15],[25,7,26],[9,1,47],[4,4,73],[0,7,100],[12,44,138],[24,82,177],[57,125,209],[134,181,229],[211,236,248],[241,233,191],[248,201,95],[255,170,0],[204,128,0],[153,87,0],[106,52,3]]
    open(filename, "w") do f
        write(f, "P3\n$width $height\n255\n")
        for i = 1:height
            for j = 1:width
                write(f,"$(MAPPING[1+pixels[i,j]][1]) $(MAPPING[1+pixels[i,j]][2]) $(MAPPING[1+pixels[i,j]][3]) ")
            end
            write(f, "\n")
        end
    end
end

function min_times(cr ::Float64, ci ::Float64, dim ::Int64, nslices ::Int64)
    tmin=0;
    
    pixels ::Matrix{Int64} = zeros(dim, dim)
    params :: Matrix{Float32} = zeros(4*nslices,1)
    for i=0:(nslices-1)
        params[4*i+1,1] = cr
        params[4*i+2,1] = ci
        params[4*i+3,1] = i*dim/nslices
        params[4*i+4,1] = dim
    end
    for i = 1:10
        t = time_ns();
        mandelbrot_with_starpu(pixels, params, nslices)
        t = time_ns()-t
        if (tmin==0 || tmin>t)
            tmin=t
        end
    end
    pixels2img(pixels,dim,dim,"out$(dim).ppm")
    return tmin
end

function display_time(cr ::Float64, ci ::Float64, start_dim ::Int64, step_dim ::Int64, stop_dim ::Int64, nslices ::Int64)
    for dim in (start_dim : step_dim : stop_dim)
        res = min_times(cr, ci, dim, nslices)
        res=res/dim/dim; # time per pixel
        println("$(dim) $(res)")
    end
end


display_time(-0.800671,-0.158392,32,32,4096,4)

@debugprint "starpu_shutdown"
starpu_shutdown()


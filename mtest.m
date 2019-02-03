
function mtest()
%% 
% Sample Matlab-Python interface
% Runs Laureline's script
% Runs our evaluation script from last semester
%
% Instructions:
% add ex_localization/ to doa-estimation/
% replace ex_localization/ex_loc_test.py with compatible version
%

% unload modules to allow reload
clear classes

%% Notes
% 1. Reloading modules:
%   For Python 2.7: use py.reload(mod)
%   For Python 3.x: use py.importlib.reload(mod)
%   Comment out the appropriate lines below depending on python version
%   verify your Python version, by running `pyversion` in MATLAB

% 2. evaluate.py (evaluation script from Dec 2018)
%   Line 9: os.unsetenv('MKL_NUM_THREADS')
%   interfacing with matlab on my machine required deleting the
%   environment variable MKL_NUM_THREADS
%
% 3. ex_localization/ex_loc_test.py
%   Line 61: os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
%   This line suppresses the warning
%   ''Your CPU supports instructions that this TensorFlow binary was not compiled
%    to use: AVX2 FMA''
%   This is just a warning saying that building TensorFlow from source can be faster
%   on this machine. It is not an error
%
% 4. The occasional crash
%    Every once in a while (very rarely) MATLAB will crash
%    Just restart MATLAB and try running it again once or twice and it will run
%
% 5. Intel MKL ERROR: Parameter 5 was incorrect on entry to DGESDD
%     This is printed when running keras.models.load_model in Laureline's code
%     from MATLAB.
%     It doesn't terminate the program, and Laureline's code still prints the
%     correct output (printed at lines 109-110 in ex_loc_test.py)
%       [[ 0. 60.]]
%       [[-63.24324324 161.05263158]]
%     TODO Verify that this is not a critical error
%     Potential solutions:
%       Build LAPACK library version from source to match MATLAB's lapack version
%       MATLAB's version is 3.7.0 on my machine
%       MATLAB version can be checked with `version -lapack` in MATLAB
%       I will try this later and see if it fixes it.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% run Laureline's code
disp([newline 'Testing Laureline''s code.'])
cd 'ex_localization'

% reload
mod_perotin = py.importlib.import_module('ex_loc_test');
py.reload(mod_perotin);
%py.importlib.reload(mod_perotin)

% run
n = py.ex_loc_test.main();
disp('Done.')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% run our evaluation code from December 2018
disp([newline 'Testing evaluation script.'])
cd '..'

% reload
mod_ours = py.importlib.import_module('evaluate');
py.reload(mod_ours);
%py.importlib.reload(mod_ours)

% run
network = 'CRNN';
lstm_out = 'Full';
out_format = 'class';
data_dirs = '../res/data';
log_dirs = '../res/log';
model_path = '../res/models/crnn_class.pth';
n = py.evaluate.test_run(network,lstm_out,out_format,data_dirs,log_dirs,model_path);
disp('Done.')

end


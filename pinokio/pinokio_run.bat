cd ..
call ..\env\Scripts\activate.bat
python -m baselines.run --alg=ppo2 --env=Pinokio-v0 --num_timesteps=2e7 --save_path=Pinokio2.pkl
pause
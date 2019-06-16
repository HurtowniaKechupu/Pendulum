import time
import gym
import numpy as np
import random
import math
env = gym.make('Pendulum-v0')

epsilon = 0.4
gamma = 1.0
alpha = 0.2

podzial_kata = 12
podzial_momentu = 4
podzial_akcji = 5

def oblicz_theta (y,x):
    if (x>0):
        return math.acos(y)
    else:
        return 2*math.pi - math.acos(y)



def aprox(number, amount,max):
    posit = int(math.floor(((number/max)+1)*amount/2))
    return posit


#Q_table = np.full((podzial_kata,podzial_momentu,podzial_akcji),0)
#Q_table = np.random.rand(podzial_kata,podzial_momentu,podzial_akcji)    #wypelnienie tabeli q losowo by bez informacji poczatkowej dzialalo
#Q_table = Q_table / 20      #podzielenie by wartości sie przypakowo nie zbiegały
Q_table = [[[4.10577263e-02,1.79307784e-02,3.15862366e-02,2.27828331e-02,7.23714357e-03],
  [3.32180655e-02,2.36544683e-02,1.70137339e-03,4.01797545e-02,4.08617602e-02],
  [3.03038299e-02,3.65378865e-02,2.12036142e-02,4.96979168e-02,2.97137783e-02],
  [4.54784087e-02,1.02088467e-03,4.54972080e-02,2.21094257e-02,1.36685741e-02]],

[[3.57421369e-02,2.16340177e-03,1.20710857e-02,2.03513884e-02,1.37475295e-02],
 [4.39213760e-02,2.16258511e-03,1.32625214e-02,2.85316331e-02,2.50406102e-02],
 [3.01388732e-02,6.55766224e-03,2.21246462e-02,2.15503444e-02,1.65347383e-02],
 [2.20212109e-02,1.00901115e-02,2.77349729e-02,1.49094783e-02,4.28080081e-02]],

[[4.07914152e-02,1.89651628e-02,3.29209510e-02,5.02176259e-04,2.65109900e-03],
 [3.58173899e-02,3.83529800e-02,3.74536775e-02,4.69452870e-02,7.84780519e-03],
 [3.72715858e-02,3.13592072e-02,2.02987009e-03,1.56100031e-02,4.93989670e-02],
 [2.35954884e-02,4.57465173e-02,4.43694224e-02,4.30913948e-02,9.54996315e-03]],

[[1.16055494e-02,7.15883445e-03,2.60399897e-02,3.56862940e-02,3.46256351e-02],
 [4.46669699e-02,1.58573040e-02,1.10219987e-02,1.98015972e-02,2.79151894e-03],
 [3.75420740e-02,2.21624384e-02,1.78835089e-04,2.77250716e-02,2.55864192e-03],
 [2.93800417e-02,1.07779140e-02,4.67092367e-02,4.87548568e-02,2.93358645e-02]],

[[2.89937871e-03,3.80876842e-03,2.70679943e-02,3.92365083e-02,4.32618606e-02],
 [1.20483614e-02,2.31281127e-02,1.68159256e-02,3.88531318e-02,3.13821647e-02],
 [2.64216072e-02,7.67280960e-03,4.11737061e-02,1.76247365e-02,4.33432964e-02],
 [2.99633841e-02,4.36643174e-02,1.67277155e-02,9.19261818e-03,6.42192958e-03]],

[[9.74130040e-03,2.52596156e-02,1.76994103e-02,4.34652555e-02,4.86769852e-02],
 [1.91740487e-02,7.11031550e-03,2.15858130e-02,4.97983373e-02,1.22772813e-02],
 [2.15230525e-02,4.62215551e-02,3.44068723e-02,4.82719812e-02,3.95665775e-02],
 [3.09910056e-02,3.32984172e-02,1.85971781e-02,3.43239856e-02,1.40017988e-02]],

[[-6.27590001e+01,-6.28729745e+01,-6.27976083e+01,-6.25834918e+01,-6.26441462e+01],
 [-6.38253193e+01,-6.37260980e+01,-6.37260650e+01,-6.38701013e+01,-6.37914696e+01],
 [-6.43795489e+01,-6.48160137e+01,-6.44382545e+01,-6.42623616e+01,-6.45269144e+01],
 [-6.34941933e+01,-6.35968670e+01,-6.39760534e+01,-6.37051377e+01,-6.40699566e+01]],

[[-6.42107171e+01,-6.37499330e+01,-6.44846137e+01,-6.42204792e+01,-6.41935064e+01],
 [-6.49558982e+01,-6.55108880e+01,-6.52837165e+01,-6.52954854e+01,-6.52775064e+01],
 [-6.63842313e+01,-6.61111642e+01,-6.61086211e+01,-6.61775308e+01,-6.62561404e+01],
 [-6.57737905e+01,-6.60253160e+01,-6.58217235e+01,-6.58543312e+01,-6.60374350e+01]],

[[-6.60475452e+01,-6.53793252e+01,-6.66572598e+01,-6.60042634e+01,-6.61364640e+01],
 [-6.78391110e+01,-6.78738937e+01,-6.70941979e+01,-6.79461923e+01,-6.78407835e+01],
 [-6.78276222e+01,-6.77661406e+01,-6.78958920e+01,-6.78580598e+01,-6.70598263e+01],
 [-6.65732267e+01,-6.66119558e+01,-6.65189025e+01,-6.64483799e+01,-6.64173715e+01]],

[[-6.69357514e+01,-6.67018046e+01,-6.69888137e+01,-6.70070746e+01,-6.69479375e+01],
 [-6.79716252e+01,-6.80651993e+01,-6.75070355e+01,-6.79529618e+01,-6.79029690e+01],
 [-6.76592494e+01,-6.76998309e+01,-6.76730330e+01,-6.65484167e+01,-6.76876230e+01],
 [-6.62577716e+01,-6.56339022e+01,-6.62358261e+01,-6.61289088e+01,-6.61530274e+01]],

[[-6.67435520e+01,-6.67848482e+01,-6.68476550e+01,-6.67612940e+01,-6.67634674e+01],
 [-6.69213442e+01,-6.66220911e+01,-6.65949008e+01,-6.68937708e+01,-6.65627518e+01],
 [-6.58512081e+01,-6.60152773e+01,-6.44271854e+01,-6.59471924e+01,-6.57882735e+01],
 [-6.45475913e+01,-6.45135663e+01,-6.46583075e+01,-6.42468341e+01,-6.45957969e+01]],

[[-6.52381840e+01,-6.45641209e+01,-6.50474476e+01,-6.44078900e+01,-6.54005001e+01],
 [-6.50696587e+01,-6.51988310e+01,-6.51520837e+01,-6.48867625e+01,-6.46033076e+01],
 [-6.30434434e+01,-6.33762455e+01,-6.32789812e+01,-6.30314647e+01,-6.34868982e+01],
 [-6.13707339e+01,-6.12426595e+01,-6.12039692e+01,-6.10946399e+01,-6.14104000e+01]]]

Q_table = np.array(Q_table)
Q_table = Q_table-70
Q_table = Q_table/70



for i_episode in range(300):
    state = env.reset() #reset środowiska

    y = state[0]
    x = state[1]
    th = oblicz_theta(y, x)
    pozycja_th = aprox(th, podzial_kata, 2 * math.pi)  # ustalenie pozycji dla kąta w tablicy q
    pozycja_momentu = aprox(state[2], podzial_momentu, 8)  # ustalenie pozycji dla momentu w tablicy q

    for t in range(100):
        env.render()    #render grafiki, chyba najlepsze miejsce



        if random.uniform(0, 1) < epsilon:
            # Check the action space
            action = env.action_space.sample()
            pozycja_akcji = aprox(action,podzial_akcji,2)   #obliczenie pozycji akcji w tabeli q
        else:
            # Check the learned values
            pozycja_akcji = np.argmax(Q_table[pozycja_th][pozycja_momentu])
            action = np.array([float(pozycja_akcji-2)/1])     #nie wiem co to -2.5/1.25 = (podzal_akcji-1/2)/((podzial_akcji-1)/4)




        next_state, reward, done, info = env.step(action)
        y = next_state[0]
        x = next_state[1]
        th = oblicz_theta(y,x)
        nowa_pozycja_th = aprox(th, podzial_kata, 2 * math.pi)
        nowa_pozycja_momentu = aprox(next_state[2], podzial_momentu, 8)

        if nowa_pozycja_momentu == podzial_momentu:             #nie mam najmniejszego pojecia dlaczego tak sie dzieje
            nowa_pozycja_momentu -= 1

        Q_value = alpha * ((reward+17)/17 + gamma * max(Q_table[nowa_pozycja_th][nowa_pozycja_momentu])- Q_table[pozycja_th][pozycja_momentu][pozycja_akcji])
        Q_table[pozycja_th][pozycja_momentu][pozycja_akcji] += Q_value

        pozycja_th = nowa_pozycja_th
        pozycja_momentu = nowa_pozycja_momentu
        print(Q_value)
        print(action,next_state,reward)



        #time.sleep(1)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

print(Q_table)

env.close()





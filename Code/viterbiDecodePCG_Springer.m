 
function [delta, psi, qt] = viterbiDecodePCG_Springer(observation_sequence, pi_vector, b_matrix, total_obs_distribution, heartrate, systolic_time, Fs,figures)

if nargin < 8
    figures = false;
end
 
springer_options = default_Springer_HSMM_options;

T = length(observation_sequence);
N = 4; % Number of states

 max_duration_D = round((1*(60/heartrate))*Fs);
 delta = ones(T+ max_duration_D-1,N)*-inf;
 psi = zeros(T+ max_duration_D-1,N);

 psi_duration =zeros(T + max_duration_D-1,N);

 observation_probs = zeros(T,N);

for n = 1:N
     pihat = mnrval(cell2mat(b_matrix(n)),observation_sequence(:,:));
    
    for t = 1:T
        
        Po_correction = mvnpdf(observation_sequence(t,:),cell2mat(total_obs_distribution(1)),cell2mat(total_obs_distribution(2)));
         
        observation_probs(t,n) = (pihat(t,2)*Po_correction)/pi_vector(n);
        
    end
end

 [d_distributions, max_S1, min_S1, max_S2, min_S2, max_systole, min_systole, max_diastole, min_diastole] = get_duration_distributions(heartrate,systolic_time);



duration_probs = zeros(N,3*Fs);
duration_sum = zeros(N,1);
for state_j = 1:N
    for d = 1:max_duration_D
        if(state_j == 1)
            duration_probs(state_j,d) = mvnpdf(d,cell2mat(d_distributions(state_j,1)),cell2mat(d_distributions(state_j,2)));
            
            if(d < min_S1 || d > max_S1)
                duration_probs(state_j,d)= realmin;
            end
            
            
        elseif(state_j==3)
            duration_probs(state_j,d) = mvnpdf(d,cell2mat(d_distributions(state_j,1)),cell2mat(d_distributions(state_j,2)));
            
            if(d < min_S2 || d > max_S2)
                duration_probs(state_j,d)= realmin;
            end
            
            
        elseif(state_j==2)
            
            duration_probs(state_j,d) = mvnpdf(d,cell2mat(d_distributions(state_j,1)),cell2mat(d_distributions(state_j,2)));
            
            if(d < min_systole|| d > max_systole)
                duration_probs(state_j,d)= realmin;
            end
            
            
        elseif (state_j==4)
            
            duration_probs(state_j,d) = mvnpdf(d,cell2mat(d_distributions(state_j,1)),cell2mat(d_distributions(state_j,2)));
            
            if(d < min_diastole ||d > max_diastole)
                duration_probs(state_j,d)= realmin;
            end
        end
    end
    duration_sum(state_j) = sum(duration_probs(state_j,:));
end


if(length(duration_probs)>3*Fs)
    duration_probs(:,(3*Fs+1):end) = [];
end

if(figures)
    figure('Name', 'Duration probabilities');
    plot(duration_probs(1,:)./ duration_sum(1),'Linewidth',2);
    hold on;
    plot(duration_probs(2,:)./ duration_sum(2),'r','Linewidth',2);
    hold on;
    plot(duration_probs(3,:)./ duration_sum(3),'g','Linewidth',2);
    hold on;
    plot(duration_probs(4,:)./ duration_sum(4),'k','Linewidth',2);
    hold on;
    legend('S1 Duration','Systolic Duration','S2 Duration','Diastolic Duration');
    pause();
end
 

qt = zeros(1,length(delta));
 
delta(1,:) = log(pi_vector) + log(observation_probs(1,:)); %first value is the probability of intially being in each state * probability of observation 1 coming from each state

 psi(1,:) = -1;

 
a_matrix = [0,1,0,0;0 0 1 0; 0 0 0 1;1 0 0 0];


 

if(springer_options.use_mex)
    
    
    [delta, psi, psi_duration] = viterbi_Springer(N,T,a_matrix,max_duration_D,delta,observation_probs,duration_probs,psi, duration_sum);
    
    
else
     
    
    for t = 2:T+ max_duration_D-1
        for j = 1:N
            for d = 1:1:max_duration_D
                 start_t = t - d;
                if(start_t<1)
                    start_t = 1;
                end
                if(start_t > T-1)
                    start_t = T-1;
                end
                 end_t = t;
                if(t>T)
                    end_t = T;
                end
                
                 [max_delta, max_index] = max(delta(start_t,:)+log(a_matrix(:,j))');
                               
                 probs = prod(observation_probs(start_t:end_t,j));
                 
                if(probs ==0)
                    probs = realmin;
                end
                emission_probs = log(probs);
                
                
                  
                if(emission_probs == 0 || isnan(emission_probs))
                    emission_probs =realmin;
                end
                
                
                 delta_temp = max_delta + (emission_probs)+ log((duration_probs(j,d)./duration_sum(j)));
                 
                
                if(delta_temp>delta(t,j))
                    delta(t,j) = delta_temp;
                    psi(t,j) = max_index;
                    psi_duration(t,j) = d;
                end
                
            end
        end
    end
end


 temp_delta = delta(T+1:end,:);
%Find the maximum value in this section, and which state it is in:
[~, idx] = max(temp_delta(:));
[pos, ~] = ind2sub(size(temp_delta), idx);

 pos = pos+T;
 
[~, state] = max(delta(pos,:),[],2);

 offset = pos;
preceding_state = psi(offset,state);

 onset = offset - psi_duration(offset,state)+1;

 qt(onset:offset) = state;

 state = preceding_state;

count = 0;
 while(onset > 2)
    
     offset = onset-1;
     preceding_state = psi(offset,state);
     
    
     onset = offset - psi_duration(offset,state)+1;
    
     
    if(onset<2)
        onset = 1;
    end
    qt(onset:offset) = state;
    state = preceding_state;
    count = count +1;
    
    if(count> 1000)
        break;
    end
end

qt = qt(1:T);



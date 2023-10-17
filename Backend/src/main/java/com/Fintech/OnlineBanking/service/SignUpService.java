package com.Fintech.OnlineBanking.service;

import java.util.HashSet;

import java.util.Optional;
import java.util.Set;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import com.Fintech.OnlineBanking.domain.UserCard;
import com.Fintech.OnlineBanking.domain.Message;
import com.Fintech.OnlineBanking.domain.PasswordsUser;
import com.Fintech.OnlineBanking.domain.StepsOfSignUp;
import com.Fintech.OnlineBanking.domain.UserInformation;
import com.Fintech.OnlineBanking.dto.UserAccountsRequest;
import com.Fintech.OnlineBanking.dto.passwordRequest;
import com.Fintech.OnlineBanking.dto.UsernameRequest;
import com.Fintech.OnlineBanking.exceptions.UnauthorizedException;
import com.Fintech.OnlineBanking.repository.UserCardRepository;
import com.Fintech.OnlineBanking.repository.UserInformationRespository;
import com.Fintech.OnlineBanking.projection.FullNameProjection;
import com.Fintech.OnlineBanking.user.Role;
import com.Fintech.OnlineBanking.user.User;
import com.Fintech.OnlineBanking.user.UserRepository;
import com.Fintech.OnlineBanking.repository.MessageRepository;
import com.Fintech.OnlineBanking.repository.PasswordsRepository;
import com.Fintech.OnlineBanking.repository.StepsOfSignUpRepository;
import com.Fintech.OnlineBanking.repository.UserAccountRepository;

@Service
public class SignUpService {
	@Autowired
	private PasswordEncoder passwordEncoder;
	@Autowired
	UserRepository userRepo;

	@Autowired
	MessageRepository messageRepo;
	@Autowired
	UserInformationRespository userInfoRepo;

	@Autowired
	UserCardRepository userVisaRepo;
	@Autowired
	UserAccountRepository userAccountRepo1;
	@Autowired
	PasswordsRepository passwords;
	@Autowired
	StepsOfSignUpRepository stepsOfSignUpRepo;

	public UserCard getCard(long NationalId, long carNumber) {
		Set<UserCard> allCard = new HashSet<UserCard>();
		try {
			allCard = userVisaRepo.findByUserInfoNationalId(carNumber);

			for (UserCard card : allCard) {
				if (card.getCardNumber() == carNumber) {
					return card;
				}
			}

		} catch (Exception ex) {

			return null;

		}
		return null;
	}

	StepsOfSignUp checkStepsOfSignUp(long nationalId) {
		StepsOfSignUp stepsOfSignUp = new StepsOfSignUp();
		try {
			stepsOfSignUp = stepsOfSignUpRepo.findByUserInfoNationalId(nationalId);

		} catch (Exception e) {

		}
		return stepsOfSignUp;
	}

	UserInformation userBlocked(long nationalId) {
		UserInformation userInfo = new UserInformation();
		try {
			userInfo = userInfoRepo.findByNationalId(nationalId);
		} catch (Exception e) {
		}

		return userInfo;
	}

	public Message findNationalVisaInfo(UserAccountsRequest user)  {
		UserInformation userInfo=userBlocked(user.getNationalId());
		if(userInfo==null) {return messageRepo.findByMessageNumber("20");}
		if(userInfo.getRemainingOpportunities()==0) {  throw new UnauthorizedException("Ddddd");/*.findByMessageNumber("13");*/}
		StepsOfSignUp stepsOfSignUp =new StepsOfSignUp();
		
		stepsOfSignUp =checkStepsOfSignUp(user.getNationalId());
		if(stepsOfSignUp!=null&&
				stepsOfSignUp.getStepOne() 
				&&
				stepsOfSignUp.getStepTwo()
				&&
				stepsOfSignUp.getStepThree()
				
			)
	           	{return messageRepo.findByMessageNumber("3");}
		if(stepsOfSignUp!=null&& stepsOfSignUp.getStepOne()) {
			return messageRepo.findByMessageNumber("61");
		}
		

		
		
		
	
	if (userInfo!=null) {
        UserCard  IdVisaInfo= userVisaRepo.findByCardNumber(user.getCardNumber());//userVisaRepository.fetVisaEnd().equals(user.getVisaEnd()));

        if(IdVisaInfo!=null) {
    
				if (
				
						(IdVisaInfo.getVisaEnd().equals( user.getCardEnd())
						
				
				
					if (IdVisaInfo.getVisaPassword() == user.getCardPassword()) {
						int RemainingOpportunities=3;

						userInfo.setRemainingOpportunities(RemainingOpportunities);
						userInfoRepo.save(userInfo);
						/*****************step one done*****************************/

						stepsOfSignUp.setStepOne(true);
						stepsOfSignUp.setUserInfo(userInfo);;
						stepsOfSignUpRepo.save(stepsOfSignUp);
						
						/**********************************************/

						
						return messageRepo.findByMessageNumber("9");//"this card and national id is vaild";

					} 
					else {

						int RemainingOpportunities=userInfo.getRemainingOpportunities() - 1;

						userInfo.setRemainingOpportunities(RemainingOpportunities);
						userInfoRepo.save(userInfo);
						
						Message m1=messageRepo.findByMessageNumber("18");
						if(userInfo.getRemainingOpportunities()>0) 
						{m1.setMessageContent("your apportunties is "+RemainingOpportunities);
						return messageRepo.findByMessageNumber("18");}
						
						else return messageRepo.findByMessageNumber("13");


					}
				} 
				
				
				
				else {						

					return messageRepo.findByMessageNumber("10");//invaild information about visa end 


				}
			}
		
		else {return messageRepo.findByMessageNumber("19");}
	}else

	{

		return messageRepo.findByMessageNumber("13");// this account is blocked contact with our services""

	}

	}

	public Message vaildUsername(UsernameRequest user) {

		UserInformation userInfo = userBlocked(user.getNationalId());
		if (userInfo == null) {
			return messageRepo.findByMessageNumber("20");
		}

		if (userInfo.getRemainingOpportunities() == 0) {
			return messageRepo.findByMessageNumber("13");
		}

		/****************** check that user do step one ***************************/
		StepsOfSignUp stepsOfSignUp = checkStepsOfSignUp(user.getNationalId());
		if (stepsOfSignUp != null && stepsOfSignUp.getStepOne() && stepsOfSignUp.getStepTwo()
				&& stepsOfSignUp.getStepThree()

		) {
			return messageRepo.findByMessageNumber("3");
		} // user already exist
		if (stepsOfSignUp != null && stepsOfSignUp.getStepOne() && stepsOfSignUp.getStepTwo()) {
			return messageRepo.findByMessageNumber("61");// user should go to the remainig operation of registeration
		}

		if (stepsOfSignUp != null && stepsOfSignUp.getStepOne()) {
			Optional<User> userFromApp = java.util.Optional.empty();
			User user2 = new User();

			try {
				userFromApp = userRepo.findByUsername(user.getUsername());

			} catch (Exception ex) {

				return messageRepo.findByMessageNumber("5");

			}
			if (userFromApp.isPresent()) {
				return messageRepo.findByMessageNumber("6");
			}
			user2.setUsername(user.getUsername());
			user2.setUserInfo(userInfo);
			userRepo.save(user2);
			/***************** step three done *****************************/

			stepsOfSignUp.setStepTwo(null);
			stepsOfSignUpRepo.save(stepsOfSignUp);
			/******************************************************************/
			return messageRepo.findByMessageNumber("7");
		}

		return messageRepo.findByMessageNumber("60");

	}

	public Message Password(passwordRequest user) {

		UserInformation userInfo = userBlocked(user.getNationalId());
		if (userInfo == null) {
			return messageRepo.findByMessageNumber("20");
		}
		if (userInfo.getRemainingOpportunities() == 0) {
			return messageRepo.findByMessageNumber("13");
		}

		StepsOfSignUp stepsOfSignUp = checkStepsOfSignUp(user.getNationalId());
		if (stepsOfSignUp != null && stepsOfSignUp.getStepOne() && stepsOfSignUp.getStepTwo()
				&& stepsOfSignUp.getStepThree()

		) {
			return messageRepo.findByMessageNumber("3");
		} // user already exist

		if (stepsOfSignUp != null && stepsOfSignUp.getStepOne() == true && stepsOfSignUp.getStepTwo() == true) {

			PasswordsUser pass = new PasswordsUser();

			User updateUser = userRepo.findByUserInfoNationalId(user.getNationalId());

			String encodePassword = passwordEncoder.encode(user.getPassword());
			updateUser.setPassword(encodePassword);
			updateUser.setPasswordChangeTime(new java.util.Date());

			pass.setPassword(encodePassword);
			pass.setUser(updateUser);
			;
			pass.setPasswordChangeTime(new java.util.Date());
			passwords.save(pass);

			// when user save his information we will save his authorization
			updateUser.setRole(Role.USER);
			userRepo.save(updateUser);
			/***************** step four done *****************************/

			stepsOfSignUp.setStepThree(true);
			stepsOfSignUpRepo.save(stepsOfSignUp);
			/**********************************************/

			return messageRepo.findByMessageNumber("7");
		}

		return messageRepo.findByMessageNumber("60");

	}

	public long loginRemainingOpportunities(String username) {
		Optional<User> user = userRepo.findByUsername(username);

		long RemainingOpportunities = user.get().getRemainingOpportunities() - 1;
		user.get().setRemainingOpportunities(RemainingOpportunities);
		if (RemainingOpportunities == 0) {
			user.get().setLocked(false);

		}
		userRepo.save(user.get());
		return RemainingOpportunities;

	}

	public UserCard SignUpUser(UserCard user, Long NationlId) {
		UserInformation userinfo = userInfoRepo.findByNationalId(NationlId);
		user.setUserInfo(userinfo);
		userVisaRepo.save(user);
		return null;
	}

	public void updatePaassword(String newPassword, User user, PasswordEncoder passwordEncoder2) {
		PasswordsUser pass = new PasswordsUser();

		String encodeedPassword = passwordEncoder2.encode(newPassword);
		user.setPassword(encodeedPassword);
		user.setPasswordChangeTime(new java.util.Date());
		pass.setPassword(encodeedPassword);
		pass.setUser(user);
		pass.setPasswordChangeTime(new java.util.Date());
		passwords.save(pass);
		userRepo.save(user);

	}

	public FullNameProjection getFullUserName(Long NationalId) {
		FullNameProjection f1 = userInfoRepo.findFullnameByNationalId(NationalId);
		return f1;
	}

	public Message isPasswordExpiredMessage(boolean passwordExpired) {
		Message m1 = messageRepo.findByMessageNumber("19");
		m1.setMessageContent("" + passwordExpired);
		return m1;
	}

}

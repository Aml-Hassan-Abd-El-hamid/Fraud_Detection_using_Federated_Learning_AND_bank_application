package com.Fintech.OnlineBanking.service;

import java.util.Collection;



import java.util.Set;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.Fintech.OnlineBanking.domain.Message;
import com.Fintech.OnlineBanking.domain.RequestCreateCard;
import com.Fintech.OnlineBanking.domain.UserAccounts;
import com.Fintech.OnlineBanking.dto.CreateRequestCardRequest;
import com.Fintech.OnlineBanking.repository.MessageRepository;
import com.Fintech.OnlineBanking.repository.RequestCreateCardRepository;
import com.Fintech.OnlineBanking.repository.UserAccountRepository;
import com.Fintech.OnlineBanking.projection.AccountTypeProjection;
import com.Fintech.OnlineBanking.user.User;

@Service
public class RequestCardService {
	
	@Autowired
	RequestCreateCardRepository requestCreateCardRepo;
	@Autowired
	MessageRepository messageRepo;


	public Message saveRequestCreateVisa(User user,CreateRequestCardRequest CreateVisaRequest) {
		

		try {
		RequestCreateCard lastRequest=	requestCreateCardRepo.findFirstByOrderByRequestNumberCreateCardDesc();
		//if(lastRequest==null) {lastRequest.setAcountNumber(0);}
		RequestCreateCard requestCreateCard=new RequestCreateCard();
		requestCreateCard.setAcountNumber(CreateVisaRequest.getAccountNumber());
		System.out.println(CreateVisaRequest.getAccountNumber());
		requestCreateCard.setCardActivation(CreateVisaRequest.getCardActivation());
		requestCreateCard.setCardName(CreateVisaRequest.getCardName());
		requestCreateCard.setCardType(CreateVisaRequest.getCardType());
		requestCreateCard.setUser(user);

		Message messageOfRequest=messageRepo.findByMessageNumber("51");
		if(lastRequest==null) {
			requestCreateCard.setRequestNumberCreateCard(1);

			messageOfRequest.setMessageContent(messageOfRequest.getMessageContent()+1);
			
		}
		else {
			requestCreateCard.setRequestNumberCreateCard(lastRequest.getRequestNumberCreateCard()+1);
                long L=lastRequest.getRequestNumberCreateCard()+1;
		messageOfRequest.setMessageContent("request Number is : "+L);
		}
		
		requestCreateCardRepo.save(requestCreateCard);

		return messageOfRequest;
		}
		catch(Exception e) {
			return messageRepo.findByMessageNumber("49");
		}
		
		
	}

	

}
